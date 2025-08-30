import modal

vol = modal.Volume.from_name("osatlas-windows-images", create_if_missing=True)

app = modal.App("download-osatlas-windows")

image = (
    modal.Image.debian_slim(python_version="3.12").uv_pip_install(
        "datasets", "huggingface_hub", "hf-transfer", "pillow", "py7zr"
    )
    # .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


@app.function(
    image=image,
    volumes={"/data": vol},
    timeout=60 * 60 * 24,
    secrets=[modal.Secret.from_name("HF-SECRET")],
)
def make_dataset():
    import io
    from pathlib import PurePosixPath
    from zipfile import ZipFile

    import datasets
    import huggingface_hub
    from PIL import Image, ImageOps
    from tqdm.auto import tqdm

    huggingface_hub.snapshot_download(
        repo_id="OS-Copilot/OS-Atlas-data",
        repo_type="dataset",
        local_dir="/data",
        allow_patterns="desktop_domain/*",
    )

    zip_path = "/data/desktop_domain/macos_images.zip"
    jpeg_bytes_list = []

    with ZipFile(zip_path) as zf:
        members = sorted(
            (i for i in zf.infolist() if not i.is_dir()), key=lambda i: i.filename
        )

        for info in tqdm(members, desc="Processing images"):
            try:
                with zf.open(info) as fp, Image.open(fp) as im:
                    im = ImageOps.exif_transpose(im)
                    im.load()
                    if im.mode not in ("RGB", "L"):
                        im = im.convert("RGB")

                    buf = io.BytesIO()
                    im.save(buf, format="JPEG", quality=85, optimize=True)

                    p = PurePosixPath(info.filename)  # path *inside* the zip
                    jpeg_bytes_list.append(
                        {
                            "image": {
                                "path": p.with_suffix(".jpg").name,
                                "bytes": buf.getvalue(),
                            }
                        }
                    )
            except Exception:
                continue

        ds = datasets.Dataset.from_list(jpeg_bytes_list).cast_column(
            "image", datasets.Image()
        )
        ds.push_to_hub("andersonbcdefg/osatlas-macos-images")
