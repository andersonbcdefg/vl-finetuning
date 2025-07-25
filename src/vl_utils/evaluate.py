import torch
from tqdm.auto import tqdm
from typing import Literal

from qwen_vl_utils import process_vision_info
from transformers import GenerationConfig  # type: ignore

from .spatial import parse_point, point_in_bbox, dist_to_center


@torch.no_grad()
def _evaluate_single(
    model,
    processor,
    dataloader,
    device,
    max_tokens: int = 25,
    format: Literal["plain", "json", "xml"] = "xml",
):
    """Evaluate on a single dataloader and return metrics."""
    model.eval()

    hits, total, dists = 0, 0, []

    for batch in tqdm(dataloader, desc="eval"):
        # ───── move tensors to GPU ─────
        gt_boxes = batch.pop("bbox")                    # (B, 4)
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        # ───── generation ─────
        generation_config = GenerationConfig(
            do_sample=False,
            max_new_tokens=max_tokens,
        )

        generation_config.temperature = None  # remove the attribute
        gen_ids = model.generate(
            **batch,
            generation_config=generation_config
        )

        # ───── trim prefixes & score sample‑wise ─────
        prompt_lens = (batch["input_ids"] != processor.tokenizer.pad_token_id).sum(-1)

        for i in range(gen_ids.size(0)):
            pred_txt = processor.tokenizer.decode(
                gen_ids[i][prompt_lens[i] :],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            point = parse_point(pred_txt, format=format)

            if point:
                if point_in_bbox(point, gt_boxes[i]):
                    hits += 1
                dists.append(dist_to_center(point, gt_boxes[i]))

            total += 1

    acc = hits / total if total else 0.0
    mean_dist = sum(dists) / len(dists) if dists else float("nan")
    return {"accuracy": acc, "mean_center_dist": mean_dist}

@torch.no_grad()
def evaluate(
    model,
    processor,
    dataloaders,
    device,
    max_tokens: int = 25,
    format: Literal["plain", "json", "xml"] = "xml",
):
    """Evaluate the model on one or more dataloaders.

    ``dataloaders`` may be either a single DataLoader or a mapping of name ->
    DataLoader. When a mapping is provided a mapping of metrics will be
    returned for each dataset name.
    """

    if isinstance(dataloaders, dict):
        results = {}
        for name, dl in dataloaders.items():
            results[name] = _evaluate_single(
                model,
                processor,
                dl,
                device,
                max_tokens=max_tokens,
                format=format,
            )
        return results
    else:
        return _evaluate_single(
            model,
            processor,
            dataloaders,
            device,
            max_tokens=max_tokens,
            format=format,
        )

# import torch
# from tqdm.auto import tqdm
# from typing import Literal

# from qwen_vl_utils import process_vision_info
# from transformers import GenerationConfig # type: ignore

# from .spatial import parse_point, point_in_bbox, dist_to_center

# @torch.no_grad()
# def evaluate(
#     model,
#     processor,
#     dataloader,
#     device,
#     max_tokens: int = 25,
#     format: Literal["plain", "json", "xml"] = "xml"
# ):
#     model.eval()

#     hits, total, dists = 0, 0, []

#     for batch in tqdm(dataloader, desc="eval"):
#         total += 1
#         gt_box = batch.pop("bbox")[0]
        # generation_config = GenerationConfig(
        #     do_sample=False,
        #     max_new_tokens=max_tokens
        # )
        # generation_config.temperature = None          # remove the attribute
#         out_ids = model.generate(
#             **batch.to(device),
#             generation_config=generation_config
#         )
#         pred_txt = processor.tokenizer.decode(
#             out_ids[0][batch["input_ids"].shape[1] :],
#             skip_special_tokens=True,
#             clean_up_tokenization_spaces=False,
#         )
#         point = parse_point(pred_txt, format=format)
#         print("predicted text:", pred_txt)
#         print("predicted point:", point)
#         print("gt_box:", gt_box)
#         if point:
#             if point_in_bbox(point, gt_box):
#                 hits += 1
#             dists.append(dist_to_center(point, gt_box))
#         del batch

#     acc = hits / total if total else 0.0
#     mean_dist = sum(dists) / len(dists) if dists else float("nan")

#     return {"accuracy": acc, "mean_center_dist": mean_dist}
