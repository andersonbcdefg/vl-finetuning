# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "modal",
# ]
# ///
import modal  # pyright: ignore

from images import vllm_image

app = modal.App("molmo-vllm")

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

MODELS_DIR = "/models"
MODEL_NAME = "moonshotai/Kimi-VL-A3B-Thinking-2506"
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
    import os
    import subprocess
    import textwrap

    # 1) Write sitecustomize that makes AutoConfig.register tolerant
    patch = textwrap.dedent("""
    try:
        from transformers.models.auto import configuration_auto as auto
        _orig_register = auto.AutoConfig.register
        def _safe_register(model_type, config, exist_ok=False):
            # force exist_ok=True to avoid ValueError on duplicates (e.g., 'aimv2')
            return _orig_register(model_type, config, exist_ok=True)
        auto.AutoConfig.register = _safe_register
    except Exception:
        # If anything changes upstream, don't block startup.
        pass
    """)
    with open("/root/sitecustomize.py", "w") as f:
        f.write(patch)

    env = os.environ.copy()
    env["PYTHONPATH"] = "/root:" + env.get("PYTHONPATH", "")

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
        str(6144),
    ]
    return subprocess.Popen(command)

    # cmd = [
    #     "vllm",
    #     "serve",
    #     "--uvicorn-log-level=info",
    #     MODEL_NAME,
    #     "--served-model-name",
    #     MODEL_NAME,
    #     "--host",
    #     "0.0.0.0",
    #     "--port",
    #     str(VLLM_PORT),
    #     "--trust-remote-code",
    #     "--api-key",
    #     API_KEY,
    # ]

    # subprocess.Popen(" ".join(cmd), shell=True)
