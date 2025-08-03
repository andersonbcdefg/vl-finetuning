from pathlib import Path
import json
import modal
import asyncio
from typing import cast

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "lm_deluge>=0.0.32", "pillow", "datasets", "torch"
)

app = modal.App("gemini-gui-labeling")

volume = modal.Volume.from_name("gemini-images", create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

MINUTES = 60

@app.function(
    image=image,
    timeout=60 * MINUTES,
    volumes={"/root/.cache/huggingface": hf_cache_vol, "/outputs": volume},
    secrets=[
        modal.Secret.from_name("HF-SECRET"),
        modal.Secret.from_name("GEMINI_PAID_API_KEY")
    ]
)
async def main():
    import datasets
    from lm_deluge import LLMClient, Conversation, Message, SamplingParams
    OUTPUT_DIR = Path("/outputs")  # Mount a volume or modal.Dataset if you wish

    ds = datasets.load_dataset(
        "andersonbcdefg/seeclick-10k-scored-yolo-annots", split="train"
    )
    ds = ds.filter(lambda x: x['num_annots'] > 2 and x['num_annots'] < 13)
    ds = ds.cast_column("image", datasets.Image(decode=False))

    print(len(ds), "images") # type: ignore

    client = LLMClient(
        model_names=["gemini-2.5-flash"],
        max_tokens_per_minute=3_000_000,
        max_concurrent_requests=50,
        sampling_params=[SamplingParams(reasoning_effort=None, max_new_tokens=4_000)]
    )
    prompt = (
        'Detect interactive UI elements (links, buttons, inputs, navs, anything you could click), '
        'with no more than 20 items. Output a json list where each entry contains the 2D bounding box '
        'in "box_2d" and an INTENT-based label of what the UI element in "label".'
        'For example: "Delete post", "Click on Amazon logo", "Navigate to home page", '
        '"Focus phone number input", "Set language to English", "Email the support team", "New to-do", etc. '
    )

    task_ids = []
    ds = cast(datasets.Dataset, ds)
    for row in ds:
        row = cast(dict, row)
        conv = Conversation([
            Message.user(prompt).add_image(row['image']['bytes'])
        ])
        task_id = client.start_nowait(conv)
        task_ids.append(task_id)
        await asyncio.sleep(0.1)

    resps = await client.wait_for_all(task_ids)

    with open(OUTPUT_DIR / "gemini-labels-seeclick.jsonl", "w") as f:
        for r in resps:
            if r and r.completion:
                f.write(json.dumps({"completion": r.completion}) + "\n")
            else:
                f.write(json.dumps({"completion": None}) + "\n")
