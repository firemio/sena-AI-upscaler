# sena-AI-upscaler

FlashVSR 2x video upscaler built on ComfyUI. Runs on **NVIDIA (CUDA)** and
**AMD (ROCm-on-Windows)** from the same scripts — the GPU vendor is auto-detected
and the encoder / device / attention settings are resolved at runtime.

## Setup

The setup script auto-detects the GPU and installs the matching PyTorch build:

```powershell
# Auto: CUDA on a GeForce box, ROCm on the AMD evo-x2 (Ryzen AI Max+ 395)
.\scripts\setup.ps1
```

Force a backend if auto-detection guesses wrong:

```powershell
.\scripts\setup.ps1 -Backend cuda        # NVIDIA (this GeForce PC)
.\scripts\setup.ps1 -Backend rocm        # AMD Strix Halo (gfx1151), default
.\scripts\setup.ps1 -Backend rocm -Gfx gfx1100   # other AMD GPU
.\scripts\setup.ps1 -Backend cpu         # no GPU (very slow)
```

> **AMD note:** the ROCm wheels come from AMD's gfx1151 index
> (`https://rocm.nightlies.amd.com/v2/gfx1151/`). If that index breaks (e.g. a
> `hipsparselt` dependency error), install scottt's self-contained Windows wheels
> instead (Python 3.12), then re-run setup which will keep them:
> ```powershell
> .\.venv312\Scripts\pip.exe install `
>   https://github.com/scottt/rocm-TheRock/releases/download/v6.5.0rc-pytorch/torch-2.7.0a0+git3f903c3-cp312-cp312-win_amd64.whl `
>   https://github.com/scottt/rocm-TheRock/releases/download/v6.5.0rc-pytorch/torchvision-0.22.0+9eb57cd-cp312-cp312-win_amd64.whl
> ```

Check what the pipeline detected:

```powershell
.\.venv312\Scripts\python.exe .\scripts\hardware.py
```

## Start ComfyUI

```powershell
# NVIDIA
.\scripts\run-comfyui.ps1 -Port 9123 *> .\comfyui-9123.out.log

# AMD (sets HIP_VISIBLE_DEVICES and clears CUDA_VISIBLE_DEVICES automatically)
.\scripts\run-comfyui.ps1 -Port 9123 -Backend rocm *> .\comfyui-9123.out.log
```

## Run the upscale

Identical command on both machines — the encoder (NVENC vs AMF vs libx264),
torch device, and SageAttention are chosen automatically:

```powershell
$env:COMFYUI_SERVER = "http://127.0.0.1:9123"
.\.venv312\Scripts\python.exe .\scripts\run_flashvsr_chunks.py `
  ".\outputs\LB熊本刑務所・刑務所前バス停　大ヒット名作　嘉門洋子　比企理恵.normalized.mkv" `
  --run-name flashvsr_no_motion_blur_run --profile balanced
```

The sharp motion profile disables the old motion-score-based profile downgrade and
uses `speed_optimization=1.5`, `quality_boost=3.0`, `stability_level=9`, and
`tile_overlap=32`.

### Overrides

| Flag | Default | Notes |
|---|---|---|
| `--encoder` | `auto` | `nvenc` / `amf` / `libx264` to force the video encoder |
| `--device` | `auto` | torch device (`auto` → cuda:0 / mps / cpu; ROCm uses cuda) |
| `--sageattention` | `auto` | only enabled on NVIDIA when the package is installed |

`FLASHVSR_VENDOR=nvidia|amd|cpu` (env var) overrides hardware auto-detection.

The final video is written to:

```text
runs\flashvsr_no_motion_blur_run\flashvsr_no_motion_blur_run.final.mp4
```

Check progress:

```powershell
(Get-Content .\runs\flashvsr_no_motion_blur_run\progress.jsonl).Count
Get-Content .\runs\flashvsr_no_motion_blur_run\progress.jsonl -Tail 5
```

## Cross-vendor notes

- **Attention:** FlashVSR falls back to PyTorch `scaled_dot_product_attention`
  (SDPA) when flash-attn / SageAttention are absent — the same path already used
  on the NVIDIA box — so AMD and NVIDIA produce consistent results.
- **Encoders:** NVENC (NVIDIA) and AMF (AMD) are added as ComfyUI VHS formats
  (`video_formats/nvenc_*.json`, `video_formats/amf_*.json`); libx264 is the CPU
  fallback used when neither GPU encoder is present in ffmpeg.
