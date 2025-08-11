import json

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "lm-deluge>=0.0.31", "huggingface_hub", "datasets", "pillow", "dotenv", "pandas"
    )
    .add_local_file("/Users/benjamin/Desktop/llm_tokens.env", "/.env")
    .add_local_file("/Users/benjamin/clicks.jsonl", "/clicks.jsonl")
)

app = modal.App("label-guiworld")

vol = modal.Volume.from_name("guiworld-ds", create_if_missing=True)

MINUTES = 60


@app.function(
    image=image, timeout=MINUTES * 60 * 12, volumes={"/root/.cache/huggingface": vol}
)
async def main():
    import pickle
    from io import BytesIO

    import dotenv
    import pandas as pd
    import tqdm
    from lm_deluge import Conversation, LLMClient, Message
    from PIL import Image

    dotenv.load_dotenv("/.env")

    # ds = datasets.load_dataset(
    #     'andersonbcdefg/guiworld-frames', split="train"
    # ).cast_column(
    #     "image", datasets.Image(decode=False)
    # )
    df = pd.read_json("/clicks.jsonl", lines=True, orient="records")

    # path2img = {}
    # for row in tqdm.tqdm(ds):
    #     path = row['image']['path']
    #     pil_img = Image.open(BytesIO(row['image']['bytes']))
    #     pil_img.thumbnail((1024, 1024))
    #     buf = BytesIO()
    #     pil_img.save(buf, format="jpeg")
    #     path2img[path] = buf.getvalue()

    # # save it
    # pickle.dump(path2img, open("/root/.cache/huggingface/path2img.pkl", "wb"))
    # print("done!")
    # return

    path2img = pickle.load(open("/root/.cache/huggingface/path2img.pkl", "rb"))
    client = LLMClient(
        model_names=[
            "claude-4-sonnet-bedrock",
            "claude-3.7-sonnet-bedrock",
            "claude-4-sonnet",
        ],
        model_weights=[0.15, 0.2, 0.65],
        max_concurrent_requests=8,
    )
    resps = []
    rows_to_keep = []
    prompts = []

    # do 500 prompts at a time
    for row in tqdm.tqdm(df.to_dict(orient="records")):
        video_id = row["video_id"]
        frame = row["frame"]
        instruction = row["instruction"]

        path = f"{video_id.split('.')[0]}_{frame}.png".replace("/", "_")
        if path not in path2img:
            print(f"missing {path}! skipping")
            continue
        rows_to_keep.append(row)
        img = path2img[path]
        pil_img = Image.open(BytesIO(img))
        w, h = pil_img.size

        prompt = Conversation(
            [
                Message.user(
                    "Identify where to click in the image to complete the instruction: "
                    + f'"{instruction}". The screen is {w}x{h}. JUST return coordinates, no yapping.'
                ).add_image(img, media_type="image/jpeg")
            ]
        )

        prompts.append(prompt)

        if len(prompts) >= 500:
            new_resps = await client.process_prompts_async(prompts)
            resps.extend(new_resps)
            prompts = []

    if prompts:
        new_resps = await client.process_prompts_async(prompts)
        resps.extend(new_resps)
        prompts = []

    # at the end, save everything
    completions = [r.completion for r in resps]
    json.dump(completions, open("/root/.cache/huggingface/completions.json", "w"))
    df_to_save = pd.DataFrame.from_records(rows_to_keep)
    df_to_save["completion"] = completions

    df_to_save.to_csv("/root/.cache/huggingface/labelled_data.csv", index=False)
