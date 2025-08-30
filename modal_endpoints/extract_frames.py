#!/usr/bin/env python
"""
Given a clicks.jsonl and a local video directory, save PNG frames to out_dir.

Videos are expected at:   videos/<video_id>.mp4
Frames saved as:          frames/<video_id>_<frame>.png
"""

import os
import modal
from collections import defaultdict
import json, cv2, pathlib, tqdm
from pathlib import Path

image = modal.Image.debian_slim(
    python_version="3.12"
).apt_install("libgl1", "libglib2.0-0").pip_install(
    "datasets", "pandas", "hf-transfer", "opencv-python"
).pip_install("pillow").pip_install("lm-deluge").env({
    "HF_HUB_ENABLE_HF_TRANSFER": "1"
}).add_local_file("/Users/benjamin/clicks.jsonl", "/clicks.jsonl")

vol = modal.Volume.from_name("gui-world", create_if_missing=True)
frames_vol = modal.Volume.from_name("gui-world-frames", create_if_missing=True)

app = modal.App("extract_frames")

MINUTES = 60

def extract_frame(cap, frame_idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, img = cap.read()
    if not ok:
        raise RuntimeError(f"cannot read frame {frame_idx}")
    return img

@app.function(
    image=image,
    volumes={
        "/root/.cache/huggingface": vol,
        "/frames": frames_vol
    },
    secrets=[modal.Secret.from_name("HF-SECRET")],
    timeout=500 * MINUTES
)
def extract_and_upload():
    import pandas as pd
    import datasets
    import huggingface_hub

    local_dir = huggingface_hub.snapshot_download(
        repo_id="ONE-Lab/GUI-World",
        repo_type="dataset"
    )

    print("downloaded to", local_dir)

    print(os.listdir(local_dir))

    OUT_DIR = pathlib.Path("/frames")
    OUT_DIR.mkdir(exist_ok=True)

    with open("/clicks.jsonl") as f:
        clicks = [json.loads(l) for l in f]

    # group by video so we open each file only once
    idxs = defaultdict(list)
    for c in clicks:
        idxs[c["video_id"]].append(c["frame"])

    for vid, frames in tqdm.tqdm(idxs.items(), desc="videos"):
        vpath = Path(local_dir) / vid
        if not vpath.exists():
            print(f"warning: {vpath} missing, skipping")
            continue
        cap = cv2.VideoCapture(str(vpath))
        for fr in frames:
            out = OUT_DIR / f'{vid.split(".")[0].replace("/", "_")}_{fr}.png'
            if out.exists():                       # don’t recompute
                continue
            try:
                img = extract_frame(cap, fr)
                cv2.imwrite(str(out), img)
            except RuntimeError as e:
                print(e)
        cap.release()

    ds = datasets.load_dataset(
        "imagefolder", data_dir="/frames"
    )

    ds.push_to_hub("andersonbcdefg/guiworld-frames")

def label():
    import pandas as pd
    import datasets
    from lm_deluge import LLMClient, Message, Conversation
    # img_dataset = datasets.load_dataset("andersonbcdefg/guiworld-frames")

    with open("/clicks.jsonl") as f:
        clicks = [json.loads(l) for l in f]

    prompts = []
    print(os.listdir("/frames"))
    for click in clicks:
        video_id = click['video_id']
        img_path = f'/frames/{video_id.replace(".", "_").replace("/", "_")}.png'
        if os.path.exists(img_path):
            pass
        else:
            print(f"image {img_path} not found")







# """
# Initial one‑time setup: create a repo and push the click JSON + frames.
# Requires `huggingface_hub>=0.20`.
# """
# from huggingface_hub import HfApi, upload_folder

# HF_TOKEN    = "hf_..."                           #  ➟  `huggingface-cli login`
# REPO_NAME   = "your-username/gui-clicks-dataset" # pick your path
# LOCAL_FILES = ["clicks.jsonl", "frames"]         # what to upload

# api = HfApi()
# # create repo if it doesn't exist
# if not api.repo_exists(REPO_NAME):
#     api.create_repo(REPO_NAME, token=HF_TOKEN, repo_type="dataset", space_sdk=None)

# # push everything (large files go through git‑LFS automatically)
# for item in LOCAL_FILES:
#     upload_folder(
#         folder_path=item,
#         repo_id=REPO_NAME,
#         path_in_repo="",      # keep original names
#         token=HF_TOKEN,
#         commit_message=f"Add {item}",
#     )
# print("✓ uploaded to https://huggingface.co/datasets/" + REPO_NAME)
