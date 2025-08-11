from typing import cast

import dotenv
from lm_deluge import Conversation
from lm_deluge.client import _LLMClient

from vl_utils.data import UIAnnotationDataset
from vl_utils.spatial import (
    Point,
    dist_to_center,
    parse_point,
    point_in_bbox,
    relative_to_absolute,
)

dotenv.load_dotenv("/Users/benjamin/Desktop/llm_tokens.env")

DEFAULT_PROMPT_TEMPLATE = (
    "Given an instruction, determine where to click in the image to complete it. "
    "Provide your action as (x,y) click coordinates.\n\nInstruction: {instruction}"
)


async def _evaluate_single(
    client: _LLMClient,  # should have "postprocess" that returns (x,y) point that can be parsed
    dataset,
    prompt_template=DEFAULT_PROMPT_TEMPLATE,
):
    task_ids = []
    bboxes = []
    sizes = []
    for i, sample in enumerate(dataset):
        image = sample["data_uri"]
        instruction = sample["instruction"]
        bbox = sample["bbox"]
        img_size = sample["size"]
        bboxes.append(bbox)
        sizes.append(img_size)

        conv = Conversation.user(
            prompt_template.format(instruction=instruction), image=image
        )
        task_id = client.start_nowait(conv)
        task_ids.append(task_id)
        if i > 10:
            break
        # break

    resps = await client.wait_for_all(task_ids)
    total, hits, dists = 0, 0, []
    for i, resp in enumerate(resps):
        pred_txt = (resp and resp.completion) or ""
        # print(pred_txt)
        point = parse_point(pred_txt, format="plain")

        bbox = bboxes[i]
        img_size = sizes[i]
        if point:
            # smart-scale point if relative
            if all([x <= 1.01 for x in point]):
                point = cast(Point, relative_to_absolute(point, img_size))
            if point_in_bbox(point, bbox):
                hits += 1
                dists.append(dist_to_center(point, bboxes[i]))
        total += 1

    acc = hits / total if total else 0.0
    mean_dist = sum(dists) / len(dists) if dists else float("nan")
    return {"accuracy": acc, "mean_center_dist": mean_dist}, resps


def evaluate(
    client,
    datasets: UIAnnotationDataset
    | list[UIAnnotationDataset]
    | dict[str, UIAnnotationDataset],
    prompt_template=DEFAULT_PROMPT_TEMPLATE,
):
    """Evaluate the model on one or more datasets"""
    client.open()
    if isinstance(datasets, dict):
        results = {}
        for name, ds in datasets.items():
            results[name] = _evaluate_single(
                client,
                ds,
                prompt_template,
            )
        return results

    elif isinstance(datasets, list):
        results = []
        for ds in datasets:
            results.append(
                _evaluate_single(
                    client,
                    ds,
                    prompt_template,
                )
            )
        return results

    else:
        return _evaluate_single(
            client,
            datasets,
            prompt_template,
        )


async def main():
    # do evaluation of kimi on webclick

    from lm_deluge import LLMClient
    from lm_deluge.models import register_model

    from vl_utils.data import load_data

    _, test = load_data("screenspot")
    register_model(
        "molmo",
        "allenai/Molmo-7B-D-0924",
        "https://taylorai--molmo-vllm-3-serve.modal.run/v1",
        "KIMI_API_KEY",
        "openai",
    )

    def _postprocess(response):
        if not response.completion:
            return response
        else:
            last_part = response.content.parts[-1]
            if last_part and last_part.text:
                text = last_part.text.strip()
                point = parse_point(text, format="xml")
                if point:
                    x, y = point
                    last_part.text = f"({x / 100}, {y / 100})"
                else:
                    point = parse_point(text, format="plain")
                    if point:
                        x, y = point
                        last_part.text = f"({x / 100}, {y / 100})"

        return response

    client = LLMClient(
        "molmo",
        postprocess=_postprocess,
        max_new_tokens=2_048,
        max_tokens_per_minute=1_000_000,
        request_timeout=250,
    )
    client.open()

    return await _evaluate_single(
        client,
        test,
        prompt_template=(
            "Given an instruction, 'point' to the 'click location'. "
            "The click location is where to click in the image to complete the instruction. "
            # "Provide your action as a pyautogui.click command.\n\n"
            "Instruction: {instruction}\n\n"
            # "Think step by step."
        ),
    )


if __name__ == "__main__":
    import asyncio
    import json

    metrics, responses = asyncio.run(main())

    completions = [res.completion for res in responses]

    json.dump(completions, open("completions.json", "w"))

    print("metrics:", metrics)
