from pathlib import Path

import torch
import modal
import pandas as pd
import datasets

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "lm_deluge", "pillow", "datasets", "torch"
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
    import os
    from typing import List, Tuple
    import datasets
    from datasets import load_dataset
    from PIL import Image, ImageDraw, ImageFont  # PIL handles alpha better than cv2
    from lm_deluge import LLMClient, Conversation, Message, SamplingParams
    OUTPUT_DIR = Path("/outputs")  # Mount a volume or modal.Dataset if you wish

    ds = load_dataset("andersonbcdefg/vibe-coded", split="train").cast_column("image", datasets.Image(decode=False))

    client = LLMClient("gemini-2.5-flash", sampling_params=[SamplingParams(reasoning_effort=None, max_new_tokens=4_000)])
    prompt = (
        'Point to the interactive UI elements, with no more than 10 items. '
        'The answer should follow the json format: [{"point": <point>, "label": <label1>}, ...]. '
        'The points are in [y, x] format normalized to 0-1000.'
    )
    prompts = []

    for row in ds:
        prompts.append(Conversation([
            Message.user(prompt).add_image(row['image']['bytes'])
        ]))

    resps = await client.process_prompts_async(prompts[:1])

    print(resps[0])

    return
