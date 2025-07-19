import json
import os

import modal
import numpy as np
from tqdm.auto import tqdm

from images import classifier_image as image

image = image.add_local_python_source("images")


class NumpyJSON(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()  # → int / float
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # → list
        return super().default(obj)


app = modal.App("seeclick-filtering")
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)


@app.function(
    image=image,
    volumes={"/root/.cache/huggingface": hf_cache_vol},
    timeout=240 * 60,
    secrets=[modal.Secret.from_name("HF-SECRET")],
)
def main():
    import datasets

    OUTPUT_DIR = "/dataset"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ANNOTATIONS = "andersonbcdefg/seeclick-web-annotations"
    IMG_DATASET = "andersonbcdefg/seeclick-10k-scored"  # repo id on the Hub

    # 1. load images
    img_ds = (
        datasets.load_dataset(IMG_DATASET, split="train")
        .cast_column("image", datasets.Image(decode=False))
        .map(lambda x: {"file_name": x["image"]["path"]})
        .select_columns(["image", "file_name", "score"])
        .cast_column("image", datasets.Image())
    )

    # 2. determine the ones to keep
    scores = img_ds["score"]
    keep_idxs = [i for i, x in enumerate(scores) if (x is not None and x < 1)]
    img_ds_hq = img_ds.select(keep_idxs)
    print("kept", len(keep_idxs))

    annots = datasets.load_dataset(ANNOTATIONS, split="train")
    ann_df = datasets.load_dataset(
        "andersonbcdefg/seeclick-web-annotations", split="train"
    ).to_pandas()

    # one row per image, “elements” is a list of dicts
    annot_df = (
        ann_df.groupby("img_filename")["elements"]
        .agg(list)  # keep duplicates, don’t flatten bbox lists
        .rename("elements")
    )

    # ── 3. join with kept‑image list ────────────────────────────────────────────────
    img_df = img_ds_hq.to_pandas()[["file_name", "score"]]
    meta_df = img_df.join(annot_df, on="file_name").dropna(subset=["elements"])

    # HF csv wants JSON strings for complex columns
    meta_df["elements"] = meta_df["elements"].apply(
        lambda lst: json.dumps(lst, cls=NumpyJSON)
    )

    # ── 4. copy images we’re keeping ────────────────────────────────────────────────
    keep = set(meta_df.file_name)
    for row in tqdm(img_ds_hq, desc="copy imgs"):
        fn = row["file_name"]
        if fn in keep:
            row["image"].save(os.path.join(OUTPUT_DIR, fn))

    # ── 5. write metadata ───────────────────────────────────────────────────────────
    meta_df.to_csv(os.path.join(OUTPUT_DIR, "metadata.csv"), index=False)

    # ── 6. reload via imagefolder (optional sanity check) ───────────────────────────
    ds = datasets.load_dataset(
        "imagefolder",
        data_dir=OUTPUT_DIR,
        split="train",
    )

    # ❺ Push to the Hub (and optionally keep a local copy).
    ds.push_to_hub(
        "andersonbcdefg/seeclick-10k-awful-q-annotated",
        max_shard_size="1GB",  # tweak if you hit the 5 GB default limit
    )
