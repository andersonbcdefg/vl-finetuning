# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "modal",
# ]
# ///
import modal  # pyright: ignore

from images import vllm_image

app = modal.App("molmo-vllm-3")

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

MODELS_DIR = "/models"
MODEL_NAME = "allenai/Molmo-7B-D-0924"
GPU_TYPE = "H100"
N_GPU = 1  # tip: for best results, first upgrade to more powerful GPUs, and only then increase GPU count
API_KEY = "super-secret-key"  # api key, for auth. for production use, replace with a modal.Secret
MINUTES = 60  # seconds
VLLM_PORT = 8000


@app.function(
    image=vllm_image,
    gpu=f"{GPU_TYPE}:{N_GPU}",
    # how long should we stay up with no requests?
    max_containers=6,
    scaledown_window=15 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
@modal.concurrent(
    max_inputs=20
)  # how many requests can one replica handle? tune carefully!
@modal.web_server(port=VLLM_PORT, startup_timeout=5 * MINUTES)
def serve():
    import subprocess

    command = [
        "vllm",
        "serve",
        "--uvicorn-log-level=info",
        MODEL_NAME,
        "--served-model-name",
        MODEL_NAME,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--trust-remote-code",
        "--enforce-eager",
        "--max-model-len",
        str(4096),
    ]
    return subprocess.Popen(command)
