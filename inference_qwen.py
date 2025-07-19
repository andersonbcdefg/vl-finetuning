# smoke test to make sure we can load model, tokenize image + text, get sensible reply

import modal

python_version = "3.11"
flash_attn_version = "2.6.3"
pytorch_version = "2.7.0"
cuda_version = "12.8.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"
flash_attn_release = "flash_attn-2.6.3+cu128torch2.7-cp311-cp311-linux_x86_64.whl"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install("git")
    .pip_install("torch==2.7.0")
    .run_commands(  # add flash-attn
        f"pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.0.9/{flash_attn_release}"
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
    .entrypoint([])
)


app = modal.App("inference-qwen25-vl")
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)


MESSAGES = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

BATCH_MESSAGES = [
    {
        "role": "system",
        "content": "you are a helpful assistant that always responds in Spanish",
    },
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://t4.ftcdn.net/jpg/02/10/96/95/360_F_210969565_cIHkcrIzRpWNZzq8eaQnYotG4pkHh0P9.jpg",
            },
            {"type": "text", "text": "what's in this image?"},
        ],
    },
]


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("HF-SECRET")],
    gpu="H100",
    volumes={"/root/.cache/huggingface": hf_cache_vol},
)
def run_inference():
    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
        device_map="auto",
    ).to("cuda")  # type: ignore
    processor = AutoProcessor.from_pretrained(model_id)

    texts = processor.apply_chat_template(
        BATCH_MESSAGES, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(BATCH_MESSAGES)  # type: ignore

    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    out = model(**inputs)
    print(out)
