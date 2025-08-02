from pathlib import Path
import json
import modal

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "lm_deluge>=0.0.28", "pillow", "datasets", "torch"
)

app = modal.App("gemini-gui-labeling")

volume = modal.Volume.from_name("gemini-images", create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

@app.function(
    image=image,
    timeout=60 * 30,
    volumes={"/root/.cache/huggingface": hf_cache_vol, "/outputs": volume},
    secrets=[
        modal.Secret.from_name("HF-SECRET"),
        modal.Secret.from_name("GEMINI_PAID_API_KEY")
    ]
)
async def main():
    from typing import List, Tuple
    import datasets
    from datasets import load_dataset
    from lm_deluge import LLMClient, Conversation, Message, SamplingParams
    OUTPUT_DIR = Path("/outputs")  # Mount a volume or modal.Dataset if you wish

    ds = load_dataset("andersonbcdefg/vibe-coded", split="train").cast_column("image", datasets.Image(decode=False))

    client = LLMClient(
        "gemini-2.5-flash",
        max_tokens_per_minute=3_000_000,
        max_concurrent_requests=50,
        sampling_params=[SamplingParams(reasoning_effort=None, max_new_tokens=4_000)]
    )
    prompt = (
        'Detect interactive UI elements (links, buttons, inputs, navs, anything you could click), '
        'with no more than 20 items. Output a json list where each entry contains the 2D bounding box '
        'in "box_2d" and a functional or visual description of what the interactive UI element '
        'does in "label". For example: "Delete post", "Amazon logo", "Navigate to home page", '
        '"Phone number input", "Set language to English", "Go to contact page", "New to-do", etc.'
    )
    prompts = []

    for row in ds:
        prompts.append(Conversation([
            Message.user(prompt).add_image(row['image']['bytes'])
        ]))

    resps = await client.process_prompts_async(prompts)

    with open(OUTPUT_DIR / "gemini-labels.jsonl", "w") as f:
        for r in resps:
            if r and r.completion:
                f.write(json.dumps({"completion": r.completion}) + "\n")
            else:
                f.write(json.dumps({"completion": None}) + "\n")
