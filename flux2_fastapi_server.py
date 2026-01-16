# /// script
# requires-python = ">=3.10,<3.11"
# dependencies = [
#   "torch==2.9.1+cu128",
#   "torchvision==0.24.1+cu128",
#   "torchaudio==2.9.1+cu128",
#   "diffusers @ git+https://github.com/huggingface/diffusers.git",
#   "transformers>=4.57.0",
#   "accelerate>=1.0.0",
#   "safetensors>=0.4.0",
#   "huggingface_hub>=0.25.0",
#   "numpy>=1.26.0",
#   "pillow>=10.0.0",
#   "fastapi>=0.115.0",
#   "uvicorn>=0.29.0",
# ]
# [tool.uv]
# extra-index-url = ["https://download.pytorch.org/whl/cu128"]
# ///

import argparse
import io
import os
import random
import time
from typing import Optional

import torch
from diffusers import Flux2KleinPipeline
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

def _resolve_model_size() -> str:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--model",
        choices=["4b", "9b"],
        default="4b",
        help="Which distilled model to load (4b or 9b)",
    )
    args, _ = parser.parse_known_args()
    return args.model


MODEL_SIZE = _resolve_model_size()

# Defaults chosen for best latency without noticeable quality loss in our tests.
DEFAULTS = {
    "ckpt": "black-forest-labs/FLUX.2-klein-4B"
    if MODEL_SIZE == "4b"
    else "black-forest-labs/FLUX.2-klein-9B",
    "num_inference_steps": 4,
    "guidance_scale": 1.0,
}

MAX_SEED = (2**31) - 1
USE_COMPILE = os.environ.get("FLUX2_COMPILE", "1") != "0"
WARMUP_RUNS = 3
WARMUP_STEPS = 4
WARMUP_PROMPT = "A high-resolution photo of a brown capybara sitting under a leafy banana tree in the rain"

app = FastAPI()
PIPE = None


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    width: int = 1024
    height: int = 1024
    num_inference_steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    seed: int = 0
    randomize_seed: bool = False


def _ensure_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this server.")


def _compile_pipe(pipe: Flux2KleinPipeline) -> bool:
    if not USE_COMPILE:
        return False
    print("Compiling model...")
    start = time.perf_counter()
    pipe.transformer = torch.compile(
        pipe.transformer,
        mode="max-autotune",
        dynamic=False,
        fullgraph=True,
    )
    pipe.vae.decode = torch.compile(
        pipe.vae.decode,
        mode="max-autotune",
        dynamic=False,
        fullgraph=True,
    )
    elapsed = time.perf_counter() - start
    print(f"Compile done in {elapsed:.2f}s")
    return True


def _warmup_pipe(pipe: Flux2KleinPipeline) -> None:
    start = time.perf_counter()
    generator = torch.Generator(device="cuda").manual_seed(0)
    for _ in range(WARMUP_RUNS):
        pipe(
            prompt=WARMUP_PROMPT,
            height=1024,
            width=1024,
            num_inference_steps=WARMUP_STEPS,
            guidance_scale=1.0,
            generator=generator,
        ).images[0]
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f"Warmup done in {elapsed:.2f}s")


def _load_pipe(ckpt: str) -> Flux2KleinPipeline:
    pipe = Flux2KleinPipeline.from_pretrained(ckpt, torch_dtype=torch.bfloat16)
    pipe.set_progress_bar_config(disable=False)
    pipe = pipe.to("cuda")
    pipe.transformer.fuse_qkv_projections()
    pipe.vae.fuse_qkv_projections()
    pipe.vae.to(memory_format=torch.channels_last)
    pipe.transformer.set_attention_backend("_native_flash")
    # compile and warmup so that first request is not affected by compilation time
    if _compile_pipe(pipe):
        _warmup_pipe(pipe)
    return pipe


@app.on_event("startup")
def _startup() -> None:
    global PIPE
    _ensure_cuda()
    PIPE = _load_pipe(DEFAULTS["ckpt"])
    print(
        "FastAPI ready. Example: curl -X POST http://localhost:6006/generate "
        "-H \"Content-Type: application/json\" "
        f"-d '{{\"prompt\":\"A wet capybara under a banana leaf, close-up photo\"}}' "
        "--output out.png"
    )


@app.get("/health")
def _health():
    return {"status": "ok"}


@app.post("/generate")
def generate(req: GenerateRequest):
    if PIPE is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    seed = req.seed
    if req.randomize_seed:
        seed = random.randint(0, MAX_SEED)

    num_inference_steps = req.num_inference_steps or DEFAULTS["num_inference_steps"]
    guidance_scale = req.guidance_scale or DEFAULTS["guidance_scale"]

    generator = torch.Generator(device="cuda").manual_seed(seed)
    image = PIPE(
        prompt=req.prompt,
        height=req.height,
        width=req.width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return Response(
        content=buf.getvalue(),
        media_type="image/png",
        headers={"X-Seed": str(seed)},
    )


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["4b", "9b"],
        default=MODEL_SIZE,
        help="Which distilled model to load (4b or 9b)",
    )
    parser.add_argument("--port", type=int, default=6006)
    args = parser.parse_args()
    MODEL_SIZE = args.model
    DEFAULTS["ckpt"] = (
        "black-forest-labs/FLUX.2-klein-4B"
        if MODEL_SIZE == "4b"
        else "black-forest-labs/FLUX.2-klein-9B"
    )
    uvicorn.run(app, host="0.0.0.0", port=args.port)
