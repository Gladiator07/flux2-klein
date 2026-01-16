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
# ]
# [tool.uv]
# extra-index-url = ["https://download.pytorch.org/whl/cu128"]
# ///

import argparse
import os
import time

import numpy as np
import torch
from diffusers import Flux2KleinPipeline
from PIL import Image

MODEL_CONFIGS = {
    "4b": {
        "ckpt": "black-forest-labs/FLUX.2-klein-4B",
        "output_suffix": "4b",
    },
    "9b": {
        "ckpt": "black-forest-labs/FLUX.2-klein-9B",
        "output_suffix": "9b",
    },
}

DEFAULT_CONFIG = {
    "num_inference_steps": 4,
    "guidance_scale": 1.0,
    "max_sequence_length": 512,
}


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Model checkpoint path",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=sorted(MODEL_CONFIGS.keys()),
        default="4b",
        help="Which distilled model to benchmark (4b or 9b)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A wet capybara under a banana leaf in heavy rain, close-up photo",
        help="Text prompt",
    )
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--guidance_scale", type=float, default=None)
    parser.add_argument("--max_sequence_length", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output image file",
    )
    parser.add_argument(
        "--fused_qkv",
        action="store_true",
        help="Enable fused qkv projections",
    )
    parser.add_argument(
        "--channels_last",
        action="store_true",
        help="Enable channels_last on VAE",
    )
    parser.add_argument(
        "--attn_backend",
        type=str,
        default=None,
        help="Attention backend (e.g., _native_flash)",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile (max-autotune, static shapes, fullgraph)",
    )
    return parser


def _set_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def _apply_defaults(args: argparse.Namespace) -> None:
    model_cfg = MODEL_CONFIGS[args.model]
    if args.ckpt is None:
        args.ckpt = model_cfg["ckpt"]
    if args.num_inference_steps is None:
        args.num_inference_steps = DEFAULT_CONFIG["num_inference_steps"]
    if args.guidance_scale is None:
        args.guidance_scale = DEFAULT_CONFIG["guidance_scale"]
    if args.max_sequence_length is None:
        args.max_sequence_length = DEFAULT_CONFIG["max_sequence_length"]
    if args.output is None:
        suffix = model_cfg["output_suffix"]
        args.output = f"bench_outputs/{suffix}/step_distilled.png"


def _infer(pipe: Flux2KleinPipeline, args: argparse.Namespace) -> "np.ndarray":
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    image = pipe(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        max_sequence_length=args.max_sequence_length,
        generator=generator,
    ).images[0]
    return np.asarray(image, dtype=np.uint8)


def main() -> None:
    args = create_parser().parse_args()
    _apply_defaults(args)

    _set_seeds(args.seed)
    pipe = Flux2KleinPipeline.from_pretrained(
        args.ckpt,
        torch_dtype=torch.bfloat16,
    )
    pipe.set_progress_bar_config(disable=False)
    pipe = pipe.to("cuda")

    if args.fused_qkv:
        pipe.transformer.fuse_qkv_projections()
        pipe.vae.fuse_qkv_projections()

    if args.channels_last:
        pipe.vae.to(memory_format=torch.channels_last)

    if args.attn_backend:
        pipe.transformer.set_attention_backend(args.attn_backend)

    if args.compile:
        compile_start = time.perf_counter()
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
        compile_end = time.perf_counter()
        print(f"Compile: {compile_end - compile_start:.2f}s")

    # warmup (first iteration includes compile time when enabled)
    warmup_times = []
    for _ in range(args.warmup):
        t0 = time.perf_counter()
        _ = _infer(pipe, args)
        t1 = time.perf_counter()
        warmup_times.append(t1 - t0)
    if warmup_times:
        print(f"Warmup[0]: {warmup_times[0]:.3f}s")

    timings = []
    last_image = None
    for _ in range(args.runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        last_image = _infer(pipe, args)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        timings.append(t1 - t0)

    timings = np.array(timings, dtype=np.float64)
    mean_s = float(timings.mean())
    std_s = float(timings.std())
    print(f"Inference: {mean_s:.3f}s +/- {std_s:.3f}s (mean +/- std over {args.runs} runs)")


    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    Image.fromarray(last_image).save(args.output)


if __name__ == "__main__":
    main()
