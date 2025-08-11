import json

import modal

app = modal.App("translate-guiworld")

prompt_template = (
    "Here is a GUI image and an instruction to be carried out in the GUI. "
    "Your job is to translate the instruction from something possibly-abstract "
    "(e.g. 'like the post', 'look at shopping cart contents', 'go home') "
    "to a purely VISUAL description that uniquely picks out what to click on. "
    "For example: 'House icon at top of screen', 'Thumbs-up button', "
    "'Event listeners tab in console', 'URL bar that says google.com'. "
    "In cases where the click target might easily be confused "
    "with similar objects, make sure to include whatever details "
    "(color, location, etc.) are needed to distinguish the target. "
    "Keep it SHORT (8-12 words) as our grounding model is not that intelligent. "
    "Don't mention every single thing surrounding the target, "
    "try to describe the target itself. Instruction: "
)

vol = modal.Volume.from_name("guiworld-ds", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "huggingface_hub", "datasets", "lm-deluge>=0.0.35", "pillow", "dotenv", "pandas"
    )
    .add_local_file("/Users/benjamin/Desktop/llm_tokens.env", "/.env")
    .add_local_file("/Users/benjamin/clicks.jsonl", "/clicks.jsonl")
)

MINUTES = 60


@app.function(
    image=image, volumes={"/root/.cache/huggingface": vol}, timeout=MINUTES * 45
)
async def translate():
    import asyncio
    import pickle

    import dotenv
    import pandas as pd
    from lm_deluge import Conversation, LLMClient, Message
    from tqdm.auto import tqdm

    dotenv.load_dotenv("/.env")
    path2img = pickle.load(open("/root/.cache/huggingface/path2img.pkl", "rb"))
    annotations = pd.read_json("/clicks.jsonl", lines=True, orient="records")

    # this part we already did
    # ds = datasets.load_dataset(
    #     'andersonbcdefg/guiworld-frames', split="train"
    # ).cast_column(
    #     "image", datasets.Image(decode=False)
    # )
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

    client = LLMClient(
        model_names="gpt-5-mini",
        max_tokens_per_minute=10_000_000,
        max_concurrent_requests=100,
        progress="manual",
    )
    client.open()
    resps = []
    rows_to_keep = []
    tasks = []

    for row in tqdm(annotations.to_dict(orient="records")):
        video_id = row["video_id"]
        frame = row["frame"]
        instruction = row["instruction"]

        path = f"{video_id.split('.')[0]}_{frame}.png".replace("/", "_")
        if path not in path2img:
            print(f"missing {path}! skipping")
            continue
        rows_to_keep.append(row)
        img = path2img[path]
        # pil_img = Image.open(io.BytesIO(img))
        # w, h = pil_img.size
        prompt = Conversation(
            [
                Message.user(prompt_template + instruction).add_image(
                    img, media_type="image/jpeg"
                )
            ]
        )

        task_id = client.start_nowait(prompt)
        tasks.append(task_id)
        await asyncio.sleep(0.01)

    resps = await client.wait_for_all(tasks)

    # at the end, save everything
    completions = [r.completion for r in resps]  # type: ignore
    json.dump(completions, open("/root/.cache/huggingface/completions2.json", "w"))
    df_to_save = pd.DataFrame.from_records(rows_to_keep)
    df_to_save["completion"] = completions

    df_to_save.to_csv("/root/.cache/huggingface/labelled_data2.csv", index=False)
