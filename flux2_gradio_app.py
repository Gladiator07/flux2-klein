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
#   "gradio>=5.50.0",
# ]
# [tool.uv]
# extra-index-url = ["https://download.pytorch.org/whl/cu128"]
# ///

import argparse
import os
import random
import time

import gradio as gr
import numpy as np
import torch
from diffusers import Flux2KleinPipeline

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
    "label": f"Distilled ({MODEL_SIZE.upper()}, 4 steps)",
    "ckpt": "black-forest-labs/FLUX.2-klein-4B"
    if MODEL_SIZE == "4b"
    else "black-forest-labs/FLUX.2-klein-9B",
    "num_inference_steps": 4,
    "guidance_scale": 1.0,
}

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 1024
USE_COMPILE = os.environ.get("FLUX2_COMPILE", "1") != "0"
WARMUP_RUNS = 3
WARMUP_STEPS = 4
WARMUP_PROMPT = "Warmup prompt to trigger compilation"


def _ensure_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this app.")


def _compile_pipe(pipe: Flux2KleinPipeline, label: str) -> bool:
    if not USE_COMPILE:
        return False
    print(f"Compiling {label}...")
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


def _warmup_pipe(pipe: Flux2KleinPipeline, label: str) -> None:
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
    print(f"Warmup done in {elapsed:.2f}s ({label})")


def _load_pipe(ckpt: str, label: str) -> Flux2KleinPipeline:
    pipe = Flux2KleinPipeline.from_pretrained(ckpt, torch_dtype=torch.bfloat16)
    pipe.set_progress_bar_config(disable=False)
    pipe = pipe.to("cuda")
    pipe.transformer.fuse_qkv_projections()
    pipe.vae.fuse_qkv_projections()
    pipe.vae.to(memory_format=torch.channels_last)
    pipe.transformer.set_attention_backend("_native_flash")
    if _compile_pipe(pipe, label):
        _warmup_pipe(pipe, label)
    return pipe


def _normalize_gallery_images(input_images):
    if not input_images:
        return None
    images = []
    for item in input_images:
        if isinstance(item, (tuple, list)) and item:
            images.append(item[0])
        else:
            images.append(item)
    return images or None


def _generate(
    prompt,
    input_images,
    seed,
    randomize_seed,
    width,
    height,
    num_inference_steps,
    guidance_scale,
):
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    pipe = PIPE
    generator = torch.Generator(device="cuda").manual_seed(seed)
    images = _normalize_gallery_images(input_images)

    if num_inference_steps is None:
        num_inference_steps = DEFAULTS["num_inference_steps"]
    if guidance_scale is None:
        guidance_scale = DEFAULTS["guidance_scale"]

    pipe_kwargs = {
        "prompt": prompt,
        "height": height,
        "width": width,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "generator": generator,
    }
    if images is not None:
        pipe_kwargs["image"] = images

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    image = pipe(**pipe_kwargs).images[0]
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return image, seed, f"Latency: {elapsed:.3f}s"


_ensure_cuda()
PIPE = _load_pipe(DEFAULTS["ckpt"], f"distilled-{MODEL_SIZE}")
print("Gradio ready at http://0.0.0.0:6006")

with gr.Blocks() as demo:
    gr.Markdown(
        f"# FLUX.2 [klein] {MODEL_SIZE.upper()}\n"
        "Distilled (4-step) variant with optimized inference."
    )
    with gr.Row():
        with gr.Column():
            prompt = gr.Text(
                label="Prompt",
                show_label=False,
                max_lines=2,
                placeholder="Enter your prompt",
            )
            run_button = gr.Button("Run")
            with gr.Accordion("Input image(s) (optional)", open=False):
                input_images = gr.Gallery(type="pil", columns=3, rows=1)
            with gr.Accordion("Advanced Settings", open=False):
                seed = gr.Slider(
                    label="Seed",
                    minimum=0,
                    maximum=MAX_SEED,
                    step=1,
                    value=0,
                )
                randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                with gr.Row():
                    width = gr.Slider(
                        label="Width",
                        minimum=256,
                        maximum=MAX_IMAGE_SIZE,
                        step=8,
                        value=1024,
                    )
                    height = gr.Slider(
                        label="Height",
                        minimum=256,
                        maximum=MAX_IMAGE_SIZE,
                        step=8,
                        value=1024,
                    )
                with gr.Row():
                    num_inference_steps = gr.Slider(
                        label="Steps",
                        minimum=1,
                        maximum=100,
                        step=1,
                        value=DEFAULTS["num_inference_steps"],
                    )
                    guidance_scale = gr.Slider(
                        label="Guidance scale",
                        minimum=0.0,
                        maximum=10.0,
                        step=0.1,
                        value=DEFAULTS["guidance_scale"],
                    )
        with gr.Column():
            result = gr.Image(label="Result", show_label=False)
            latency = gr.Markdown(value="Latency: -")

    gr.on(
        triggers=[run_button.click, prompt.submit],
        fn=_generate,
        inputs=[
            prompt,
            input_images,
            seed,
            randomize_seed,
            width,
            height,
            num_inference_steps,
            guidance_scale,
        ],
        outputs=[result, seed, latency],
    )

demo.launch(server_name="0.0.0.0", server_port=6006)
