# FLUX.2 klein distilled (4B and 9B)

Minimal, optimized inference for [FLUX.2 klein](https://bfl.ai/blog/flux2-klein-towards-interactive-visual-intelligence).

## Benchmarks (A100, 1024x1024)

### 4B distilled

| Config | Mean (s) | Std (s) | Speedup vs baseline |
| --- | --- | --- | --- |
| bf16 baseline | 1.216 | 0.002 | 1.00x |
| bf16 + torch compile | 0.928 | 0.006 | 1.31x |
| bf16 + torch compile + fused_qkv + channels_last + native_flash | 0.896 | 0.004 | 1.36x |

### 9B distilled

| Config | Mean (s) | Std (s) | Speedup vs baseline |
| --- | --- | --- | --- |
| bf16 baseline | 2.240 | 0.008 | 1.00x |
| bf16 + torch compile | 1.815 | 0.008 | 1.23x |
| bf16 + torch compile + fused_qkv + channels_last + native_flash | 1.787 | 0.007 | 1.25x |

## Quick start

All scripts use uv inline script metadata (PEP 723), so dependencies are installed automatically when you run them.

Set your Hugging Face token and run.

```bash
export HF_TOKEN=YOUR_TOKEN
```

Gradio UI:

```bash
uv run flux2_gradio_app.py --model 4b
```

FastAPI server:

```bash
uv run flux2_fastapi_server.py --model 4b
```

The FastAPI server listens on port 6006. Send a request:

```bash
curl -X POST http://localhost:6006/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"A wet capybara under a banana leaf, close-up photo"}' \
  --output out.png
```

To switch to 9B, pass `--model 9b` to either script.

## Benchmarking

`benchmark.py` also uses inline script metadata, so uv installs the dependencies automatically.

```bash
uv run benchmark.py --model 4b --compile --fused_qkv --channels_last --attn_backend _native_flash
```

If you want the baseline run:

```bash
uv run benchmark.py --model 4b
```
