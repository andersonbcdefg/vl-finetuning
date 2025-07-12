import random
from functools import partial
from math import sqrt
from pathlib import Path

import modal
import torch
from datasets import concatenate_datasets
from torch.optim import AdamW
from tqdm import tqdm
from typing import Literal

from vl_utils.loss import build_ntl_index, compute_ntl_loss
from vl_utils.data import strip_null_images, convert_to_messages
from vl_utils.spatial import parse_point, point_in_bbox, dist_to_center

# ------------------------------------------------------------------------- #
#   Modal image: system packages + Python deps                              #
# ------------------------------------------------------------------------- #
python_version = "3.11"
flash_attn_version = "2.6.3"
pytorch_version = "2.7.1"
cuda_version = "12.8.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"
flash_attn_release = "flash_attn-2.6.3+cu128torch2.7-cp311-cp311-linux_x86_64.whl"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install("git")
    .pip_install("torch==2.7.1")
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
    .pip_install(
        "cut-cross-entropy @ git+https://github.com/apple/ml-cross-entropy.git"
    )
    .pip_install("bitsandbytes")
    .pip_install_private_repos(
        "github.com/andersonbcdefg/vl-finetuning.git",
        git_user="andersonbcdefg",
        secrets=[
            modal.Secret.from_name("my-github-secret")
        ]
    )
    .entrypoint([])
)

app = modal.App("finetune-qwen25-vl")
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

# ------------------------------------------------------------------------- #
# 2️⃣ / 3️⃣  Spatial/point helpers
# ------------------------------------------------------------------------- #
def _strip_answer(conv: list[dict]) -> list[dict]:
    """Remove the final assistant message so the model must predict it."""
    if conv and conv[-1]["role"] == "assistant":
        return conv[:-1]
    return conv

@torch.no_grad()
def evaluate(model, processor, dataset, device, max_tokens: int = 20, format: Literal["plain", "json", "xml"] = "xml"):
    from qwen_vl_utils import process_vision_info

    model.eval()

    hits, total, dists = 0, 0, []

    for item in tqdm(dataset, desc="eval"):
        total += 1
        conv = item["messages"]
        gt_box = item["bbox"]  # (x1, y1, x2, y2)

        # build inputs just like generation example
        conv = strip_null_images(conv)
        conv = _strip_answer(conv)
        prompt = processor.apply_chat_template(
            conv, add_generation_prompt=True, tokenize=False
        )
        images, videos = process_vision_info(conv)  # type: ignore

        inputs = processor(
            text=[prompt],
            images=images,
            videos=videos,
            padding=True,
            return_tensors="pt",
        ).to(device)

        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,  # turn off sampling
            temperature=0.0,  # no randomness even sampling
        )
        pred_txt = processor.tokenizer.decode(
            out_ids[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        point = parse_point(pred_txt, format=format)
        if point:
            if point_in_bbox(point, gt_box):
                hits += 1
            dists.append(dist_to_center(point, gt_box))

    acc = hits / total if total else 0.0
    mean_dist = sum(dists) / len(dists) if dists else float("nan")
    return {"accuracy": acc, "mean_center_dist": mean_dist}


# ------------------------------------------------------------------------- #
#                              Training entry-point                         #
# ------------------------------------------------------------------------- #
MINUTES = 60


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("HF-SECRET")],
    gpu="H100",
    volumes={"/root/.cache/huggingface": hf_cache_vol},
    timeout=240 * MINUTES,
)
def train():
    import math

    import torch
    from cut_cross_entropy import linear_cross_entropy  # type: ignore
    from datasets import load_dataset
    from qwen_vl_utils import process_vision_info
    from torch.nn.utils import clip_grad_norm_
    from torch.optim.lr_scheduler import LambdaLR
    from torch.utils.data import DataLoader

    # from torchao.optim import AdamW8bit
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    # --------------------------- config -----------------------------------
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    dataset_id = "Hcompany/WebClick"
    groundui_id = "agent-studio/GroundUI-18K"
    out_dir = Path("/root/checkpoints/qwen").expanduser()
    epochs = 6
    per_gpu_bs = 2
    grad_accum = 8
    lr = 2e-5
    warmup_frac = 0.05
    wd = 0.01
    max_grad_norm = 1.0
    dtype = torch.bfloat16
    device = torch.device("cuda")
    use_ntl_loss = True
    ntl_loss_coeff = 0.3

    # ---------------------------- data ------------------------------------
    processor = AutoProcessor.from_pretrained(
        model_id, trust_remote_code=True
    )  # , min_pixels=min_pixels, max_pixels=max_pixels)
    processor.tokenizer.padding_side = "right"
    ds1 = (
        load_dataset(dataset_id, split="test")
        .map(
            partial(convert_to_messages, format="xml"),
            batched=True,
            batch_size=32,
            num_proc=8,  # type: ignore
            load_from_cache_file=False,  # type: ignore
        )
        .select_columns(["messages", "bbox"])
    )
    ds2 = (
        load_dataset(groundui_id, split="train")
        .map(
            partial(convert_to_messages, bbox_type="absolute", format="xml"),
            batched=True,
            batch_size=32,
            num_proc=8,  # type: ignore
            load_from_cache_file=False,  # type: ignore
        )
        .select_columns(["messages", "bbox"])
    )
    ds = concatenate_datasets([ds1, ds2])  # type: ignore
    random.seed(42)
    test_ids = random.sample(range(len(ds)), 100)
    train = ds.select([x for x in range(len(ds)) if x not in test_ids])  # type: ignore
    test = ds.select(test_ids)  # type: ignore

    def collate_fn(batch):
        conversations = [strip_null_images(item["messages"]) for item in batch]

        texts = [
            processor.apply_chat_template(c, tokenize=False) for c in conversations
        ]

        image_inputs, _ = process_vision_info(conversations)  # type: ignore

        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=None,
            padding=True,
            return_tensors="pt",
        )

        labels = inputs["input_ids"].clone()

        # compute loss only on completion
        IM_START = 151644
        ASSISTANT = 77091

        for i, ids in enumerate(inputs["input_ids"]):
            # locate the last "<|im_start|> assistant" pair – that’s the answer we want
            starts = (ids == IM_START).nonzero(as_tuple=True)[0]
            tgt_start = None
            for s in reversed(starts.tolist()):
                if s + 1 < len(ids) and ids[s + 1] == ASSISTANT:
                    tgt_start = s + 2  # first token after “assistant”
                    break
            if tgt_start is None:
                raise ValueError("no assistant response found in input")
            else:
                labels[i, :tgt_start] = -100  # ignore system/user & image tokens

        inputs["labels"] = labels
        return inputs

    train_dl = DataLoader(
        train,  # type: ignore
        batch_size=per_gpu_bs,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # ---------------------------- model -----------------------------------
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
        device_map="auto",
    ).to(device)  # type: ignore
    model.train()

    backbone = model.model  # same params, no LM head
    lm_weight = model.lm_head.weight  # shared weight matrix (V, H)

    num_ids, num_vals, token_id_to_val = build_ntl_index(
        processor.tokenizer, lm_weight.size(0)
    )

    # ----------------------- optimiser & sched ---------------------------
    # opt = AdamW8bit(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=wd)
    opt = AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=wd)

    total_steps = math.ceil(len(train_dl) / grad_accum) * epochs
    warmup_steps = int(total_steps * warmup_frac)
    steps_so_far = 0

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 1.0 - prog)

    sched = LambdaLR(opt, lr_lambda)

    # do initial eval
    results = evaluate(model, processor, test, device)
    print(
        f"📊 accuracy: {results['accuracy']:.3%} | "
        f"avg centre-distance: {results['mean_center_dist']:.4f}"
    )

    # --------------------------- training ---------------------------------
    for ep in range(epochs):
        # pbar = tqdm(dl, desc=f"epoch {ep + 1}/{epochs}", leave=False)
        opt.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_dl, 1):
            steps_so_far += 1
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=dtype):
                labels = batch.pop("labels")

                # forward the trunk
                last_hidden = backbone(
                    **batch,
                    use_cache=False,  # no KV cache during training
                    return_dict=True,
                ).last_hidden_state

                # CCE handles the causal shift internally
                ce_loss = linear_cross_entropy(
                    last_hidden,  # (B, L, H)
                    lm_weight,  # (V, H)
                    labels,  # (B, L)
                    shift=1,  # predict next token
                    ignore_index=-100,  # same semantics as F.cross_entropy
                )

                if use_ntl_loss:
                    ntl_loss = ntl_loss_coeff * compute_ntl_loss(
                        num_ids,
                        num_vals,
                        token_id_to_val,
                        last_hidden,
                        labels,
                        lm_weight,
                    )
                else:
                    ntl_loss = torch.tensor(
                        0.0, dtype=ce_loss.dtype, device=ce_loss.device
                    )

                normalized_loss = (ce_loss + ntl_loss) / grad_accum

            normalized_loss.backward()

            if step % grad_accum == 0:
                print(
                    f"Step: {step}/{len(train) // per_gpu_bs}, CE: {ce_loss.item():.3f}, NTL: {ntl_loss.item()}"
                )
                clip_grad_norm_(model.parameters(), max_grad_norm)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)

            # ------------- checkpoint ----------------------------------------
            if steps_so_far % 1000 == 0:
                save_path = out_dir / f"step-{steps_so_far}"
                save_path.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(save_path, safe_serialization=True)
                processor.save_pretrained(save_path)
                print(f"✅ saved {save_path}")

                # ------------- evaluate ----------------------------------------
                results = evaluate(model, processor, test, device)
                print(
                    f"📊 accuracy: {results['accuracy']:.3%} | "
                    f"avg centre-distance: {results['mean_center_dist']:.4f}"
                )

    save_path = out_dir / "final_ckpt"
    save_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_path, safe_serialization=True)
    processor.save_pretrained(save_path)
    print(f"✅ saved {save_path}")

    # ------------- evaluate ----------------------------------------
    results = evaluate(model, processor, test, device)
    print(
        f"📊 accuracy: {results['accuracy']:.3%} | "
        f"avg centre-distance: {results['mean_center_dist']:.4f}"
    )

    print("🎉 training complete")
