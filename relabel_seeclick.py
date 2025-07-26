import glob
import json
import io, math
import asyncio
from typing import Iterable
import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "build-essential", "pkg-config", "git",

        # Core Pillow dependencies
        "libjpeg-dev",          # JPEG
        "zlib1g-dev",           # PNG
        "libtiff-dev",          # TIFF
        "libfreetype6-dev",     # Freetype
        "liblcms2-dev",         # LittleCMS
        "libwebp-dev",          # WebP
        "libopenjp2-7-dev",     # JPEG‑2000 (openjpeg)
        "libimagequant-dev",    # ImageQuant
        "libavif-dev",          # AVIF

        # Raqm stack for complex text
        "libraqm-dev", "libfribidi-dev", "libharfbuzz-dev",

        # Misc support libs
        "libxcb1-dev"
    )
    .pip_install("torch", "datasets", "pillow-simd", "lm-deluge>=0.0.26", "tqdm")
    .add_local_python_source("vl_utils")
)

app = modal.App("relabel-seeclick")
vol = modal.Volume.from_name("seeclick-instructions", create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

@app.function(
    image=image,
    cpu=1.0,
    volumes={"/output": vol, "/root/.cache/huggingface": hf_cache_vol},
    secrets=[modal.Secret.from_name("OPENAI_FREE_SHARE_DATA_API_KEY")],
    timeout=60 * 500
)
async def main():
    from tqdm.auto import tqdm
    import datasets
    from PIL import Image
    from lm_deluge import Conversation, Message, LLMClient
    from vl_utils.draw import draw_overlay, annotate_many

    ds = datasets.load_dataset(
        "andersonbcdefg/seeclick-10k-scored-yolo-annots", split="train"
    )
    ds = ds.filter(lambda x: x['num_annots'] > 2 and x['num_annots'] < 13)
    ds = ds.cast_column("image", datasets.Image(decode=False))
    print(len(ds), "images") # type: ignore

    prompt = (
        "In the image, there is a bounding box drawn, with a big arrow pointing to it. "
        "Write an instruction that would get someone to click on the interactive "
        "element in the bounding box, assuming they can't see the box/arrow. "
        "Example instructions: "
        '"Navigate to the Home page", "Click on the first blog post", "Like the post", "Focus the address field", '
        '"Click the right arrow in the carousel", etc. '
        "IMPORTANT: If the box is too small, you can't see what it contains, or it's just full of empty space, "
        "just reply [EMPTY] so we know to remove it from the dataset."
    )

    prompts = []
    client = LLMClient(
        "gpt-4.1-mini",
        max_concurrent_requests=100,
        max_tokens_per_minute=10_000_000
    )
    # for row_idx, row in tqdm(enumerate(ds), total=len(ds)):
    #     base_im = Image.open(io.BytesIO(row['image']['bytes'])).convert("RGB")
    #     imgs = annotate_many(base_im, [elem['bbox'] for elem in row['seeclick_elements']])
    #     for img_idx, img_bytes in enumerate(imgs):
    #         with open(f"/output/{row_idx}-{img_idx}.jpeg", "wb") as f:
    #             f.write(img_bytes)
    for img_path in glob.glob("/output/*.jpeg"):
        prompts.append(
            Conversation([Message.user(prompt).add_image(img_path)])
        )

    resps = await client.process_prompts_async(prompts)
    with open("/output/responses.jsonl", "w") as f:
        for resp in tqdm(resps):
            if resp and resp.completion:
                f.write(json.dumps({
                    "completion": resp.completion
                }) + "\n")
            else:
                f.write(json.dumps({
                    "completion": None
                }) + "\n")
    #     if len(prompts) > 3000:
    #         print("starting a batch!")
    #         tasks.append(
    # batch_ids = await client.submit_batch_job(
    #     prompts,
    #     batch_size=5_000
    # )
    # print(batch_ids)
    #         prompts = []



    # insert back into the dataset
