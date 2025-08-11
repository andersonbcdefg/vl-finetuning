import os

from lm_deluge import Conversation, LLMClient, Message
from lm_deluge.models import register_model

os.environ["KIMI_API_KEY"] = "super-secret-key"

register_model(
    "kimi",
    "moonshotai/Kimi-VL-A3B-Thinking-2506",
    "https://taylorai--molmo-vllm-serve.modal.run/v1",
    "KIMI_API_KEY",
    "openai",
)

prompt = Conversation(
    [
        Message.user("click on July 5").add_image(
            "/Users/benjamin/Downloads/planit_with_cursor_314_254.png"
        )
    ]
)

client = LLMClient("kimi", request_timeout=120)

res = client.process_prompts_sync([prompt])

print(res[0].completion)  # type: ignore
