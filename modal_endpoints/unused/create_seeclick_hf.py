import pathlib
import subprocess

import modal

from images import classifier_image as image

image = image.apt_install("aria2").add_local_python_source("images")

app = modal.App("put-seeclick-on-huggingface")
vol = modal.Volume.from_name("seeclick-web-270k", create_if_missing=True)


@app.function(image=image, volumes={"/images": vol}, timeout=240 * 60)
def main():
    # we need to start by downloading this gigantic ZIP file
    ZIP_PATH = "https://box.nju.edu.cn/seafhttp/files/d5aea625-ea1a-40e3-9530-7a2f42ea5d1c/seeclick_web_imgs.zip"
    TARGET = pathlib.Path("/images/big_archive.zip")

    cmd = [
        "aria2c",
        "-c",  # resume if interrupted
        "-s1",
        "-x1",  # server doesn’t advertise Accept‑Ranges
        "--file-allocation=none",  # avoid pre‑allocating 128 GB
        "--retry-wait=30",  # back‑off between attempts
        "--max-tries=0",  # keep retrying until done
        "-o",
        str(TARGET),  # output file
        ZIP_PATH,
    ]

    subprocess.check_call(cmd)
