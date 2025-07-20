import json
import time
from pathlib import Path
import modal

# ------------------------------------------------------------------------- #
#   Modal image: system packages + Python deps                              #
# ------------------------------------------------------------------------- #
from images import qwen_image as image
from vl_utils.data import load_data, create_dataloader

# from vl_utils.data_old import load_data, collate
from functools import partial
from vl_utils.evaluate import evaluate
from vl_utils import freeze_layers

app = modal.App("finetune-qwen25-vl")
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
metrics_vol = modal.Volume.from_name("vl-ft-metrics", create_if_missing=True)

MINUTES = 60

def _print_metrics(results: dict):
  for name, metrics in results.items():
        print(
            f"[{name}] 📊 accuracy: {metrics['accuracy']:.3%} | "
            f"avg centre-distance: {metrics['mean_center_dist']:.4f}"
        )
        eval_metrics.append(
            {
                "dataset": name,
                "step": steps_so_far,
                "accuracy": metrics["accuracy"],
                "mean_center_dist": metrics["mean_center_dist"],
            }
        )

@app.function(
    image=image,
    secrets=[modal.Secret.from_name("HF-SECRET")],
    gpu="H100",
    volumes={"/root/.cache/huggingface": hf_cache_vol, "/metrics": metrics_vol},
    timeout=240 * MINUTES,
)
def train(run_name: str):
    import bitsandbytes as bnb
    import torch
    from cut_cross_entropy import linear_cross_entropy  # type: ignore
    from liger_kernel.transformers import apply_liger_kernel_to_qwen2_5_vl # type: ignore
    from torch.nn.utils import clip_grad_norm_
    from torch.optim.lr_scheduler import LambdaLR
    from torch.utils.data import DataLoader
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    apply_liger_kernel_to_qwen2_5_vl()

    # --------------------------- config -----------------------------------
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    out_dir = Path("/root/checkpoints/qwen").expanduser()
    per_gpu_bs = 4
    grad_accum = 4
    lr = 2e-5
    wd = 0.01
    max_grad_norm = 1.0
    dtype = torch.bfloat16
    device = torch.device("cuda")
    warmup_steps = 100
    total_steps = 2_000
    train_dataset = "seeclick-5"
    eval_datasets = ["webclick", "screenspot"]
    test_size = 125
    test_dataset = "screenspot"
    eval_every = 400
    test_size = 250
    seed = 42

    # ---------------------------- data ------------------------------------
    train_processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    train_processor.tokenizer.padding_side = "right"

    # evaluation (left‑pad so we can batch)
    eval_processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    eval_processor.tokenizer.padding_side = "left"
    eval_processor.tokenizer.pad_token = eval_processor.tokenizer.eos_token  # safety

    # Load training dataset and evaluation datasets separately
    eval_dataloaders = {}
    train, test_split = load_data(train_dataset, test_size, seed)
    
    # always include test split of the train data in the loader
    eval_dataloaders["held_out_train"] = create_dataloader(
      test_split, eval_processor, batch_size=per_gpu_bs * 2, num_workers=4, eval=True
    )

    
    for name in eval_datasets:
        _, ds = load_data(name, test_size, seed)
        if ds is not None:
            eval_dataloaders[name] = create_dataloader(
                ds, eval_processor, batch_size=per_gpu_bs * 2, num_workers=4, eval=True
            )

    train_dl = create_dataloader(
        train, train_processor, batch_size=per_gpu_bs, num_workers=4, eval=False
    )


    # ---------------------------- model -----------------------------------
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
        device_map="auto",
    )  # .to(device)  # type: ignore
    model.train()

    print("memory after loading model:")
    print(torch.cuda.memory_allocated() / 1e9)

    # save some memory
    freeze_layers(model, ["visual.patch_embed", "visual.blocks"])

    backbone = model.model  # same params, no LM head
    lm_weight = model.lm_head.weight  # shared weight matrix (V, H)

    # ----------------------- optimiser & sched ---------------------------
    # opt = AdamW8bit(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=wd)
    opt = bnb.optim.AdamW8bit(
        model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=wd
    )
    steps_so_far = 0

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 1.0 - prog)

    sched = LambdaLR(opt, lr_lambda)

    train_metrics = []
    eval_metrics = []

    # initial eval
    results = evaluate(model, eval_processor, eval_dataloaders, device)
    _print_metrics(results)


    # --------------------------- training ---------------------------------
    last_step = time.time()
    epoch = 0
    while True:
        if steps_so_far >= total_steps:
            break
        epoch += 1
        print(f"=== Epoch {epoch} ===")
        opt.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_dl, 1):
            steps_so_far += 1
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            # print("shape:", batch['input_ids'].shape)
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
                train_metrics.append({"step": steps_so_far, "loss": ce_loss.item()})

                normalized_loss = ce_loss / grad_accum  #  + ntl_loss) / grad_accum

            normalized_loss.backward()

            if steps_so_far % grad_accum == 0:
                now = time.time()
                step_time = now - last_step
                last_step = now
                print(
                    f"Step: {steps_so_far}/{total_steps}, CE: {ce_loss.item():.3f}, Time: {step_time:.2f}s, LR: {sched.get_last_lr()[0]:.4f}"
                )
                clip_grad_norm_(model.parameters(), max_grad_norm)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)

            # ------------- checkpoint ----------------------------------------
            if steps_so_far % eval_every == 0:
                save_path = out_dir / f"step-{steps_so_far}"
                save_path.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(save_path, safe_serialization=True)
                # processor.save_pretrained(save_path)
                print(f"✅ saved {save_path}")

                
                # ------------- evaluate ---------------------------------------
                results = evaluate(model, eval_processor, eval_dataloaders, device)
                _print_metrics(results)
                
                last_step = (
                    time.time()
                )  # shouldn't be abnormally long step time after eval


            if steps_so_far >= total_steps:
                break

    save_path = out_dir / "final_ckpt"
    save_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_path, safe_serialization=True)
    print(f"✅ saved {save_path}")
    
    # ------------- evaluate ---------------------------------------
    results = evaluate(model, processor, eval_dataloaders, device)
    _print_metrics(results)

    print("saving metrics...")
    out_dir = f"/metrics/{run_name}"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/train.json", "w") as f:
        json.dump(train_metrics, f)

    with open(f"{out_dir}/eval.json", "w") as f:
        json.dump(eval_metrics, f)

    print("🎉 training complete")
