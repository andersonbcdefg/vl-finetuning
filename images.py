import modal

python_version = "3.11"
flash_attn_version = "2.7.4"
pytorch_version = "2.7.1"
cuda_version = "12.8.1"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"
FLASH_ATTN_URL = "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.3.10/flash_attn-2.7.4+cu128torch2.7-cp311-cp311-linux_x86_64.whl"

base_image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install("git")
    .pip_install("torch==2.7.1")
)

yolo_image = (
    base_image.apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "transformers",
        "datasets",
        "torchvision",
        "accelerate",
        "datasets",
        "trl",
        "peft",
        "timm",
        "pillow",
        "hf_xet",
        "torchao",
        "open_clip_torch",
        "pillow",
        "bitsandbytes",
        "lm-deluge",
        "ultralytics",
        "opencv-python",
        "imagehash",
    )
    .pip_install("qwen-vl-utils")
    .entrypoint([])
    .add_local_file("model_v1_5.pt", "/yolo_model.pt")
    .add_local_python_source("images")
)

classifier_image = (
    base_image.run_commands(  # add flash-attn
        f"pip install {FLASH_ATTN_URL}"
    )
    .pip_install(
        "transformers",
        "datasets",
        "torchvision",
        "qwen-vl-utils",
        "reportlab",
        "matplotlib",
        "numpy",
        "accelerate",
        "datasets",
        "trl",
        "peft",
        "timm",
        "pillow",
        "hf_xet",
        "torchao",
        "open_clip_torch",
        "pillow",
        "bitsandbytes",
    )
    .entrypoint([])
    .add_local_python_source("images")
)

qwen_image = (
    base_image.run_commands(  # add flash-attn
        f"pip install {FLASH_ATTN_URL}"
    )
    .pip_install(
        "transformers",
        "datasets",
        "torchvision",
        "qwen-vl-utils",
        "reportlab",
        "matplotlib",
        "numpy",
        "accelerate",
        "datasets",
        "trl",
        "peft",
        "timm",
        "pillow",
        "hf_xet",
        "torchao",
    )
    .pip_install(
        "cut-cross-entropy @ git+https://github.com/apple/ml-cross-entropy.git"
    )
    .pip_install("bitsandbytes", "liger-kernel")
    # .pip_install_private_repos(
    #     "github.com/andersonbcdefg/vl-finetuning.git@f5c39fb",
    #     git_user="andersonbcdefg",
    #     secrets=[modal.Secret.from_name("my-github-secret")],
    # )
    .entrypoint([])
    .add_local_python_source("images")
    .add_local_python_source("vl_utils")
)

vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm==0.9.1",
        "huggingface_hub[hf_transfer]==0.32.0",
        "flashinfer-python==0.2.6.post1",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        "tensorflow",
        "transformers>=4.53.0",
        "blobfile",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_commands(  # add flash-attn
        f"pip install {FLASH_ATTN_URL}"
    )
    .add_local_python_source("images")  # faster model transfers
)
