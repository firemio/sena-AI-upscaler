"""GPU backend detection for the FlashVSR upscaler.

Auto-detects whether the host has an NVIDIA (CUDA), AMD (ROCm/HIP) or no GPU and
resolves the matching ffmpeg encoder, ComfyUI/VHS output format, SageAttention
setting and torch device. This lets the exact same pipeline run unchanged on the
NVIDIA workstation (GeForce) and on the AMD evo-x2 (Ryzen AI Max+ 395 / Radeon
8060S, gfx1151) without editing the run scripts.

Notes for AMD:
  * ROCm builds of PyTorch expose the HIP backend through the ``torch.cuda.*``
    API, so ``device="auto"`` / ``cuda:0`` works there too.
  * SageAttention (Triton/CUDA only) is force-disabled on non-NVIDIA hardware.

Override the detected vendor with the ``FLASHVSR_VENDOR`` env var
(``nvidia`` / ``amd`` / ``cpu``) when auto-detection guesses wrong.
"""
from __future__ import annotations

import functools
import json
import os
import shutil
import subprocess

VENDORS = ("nvidia", "amd", "cpu")

# vendor -> (ffmpeg codec, VHS output format)
_ENCODER_TABLE = {
    "nvidia": ("h264_nvenc", "video/nvenc_h264-mp4"),
    "amd": ("h264_amf", "video/amf_h264-mp4"),
    "cpu": ("libx264", "video/h264-mp4"),
}
_CODEC_TO_FORMAT = {
    "h264_nvenc": "video/nvenc_h264-mp4",
    "h264_amf": "video/amf_h264-mp4",
    "libx264": "video/h264-mp4",
}


def _ffmpeg_path() -> str:
    return (
        shutil.which("ffmpeg")
        or os.environ.get("FFMPEG", "")
        or r"C:\Users\starc\AppData\Local\Microsoft\WinGet\Links\ffmpeg.exe"
    )


def _windows_gpu_names() -> str:
    if os.name != "nt":
        return ""
    try:
        return subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_VideoController | "
                "Select-Object -ExpandProperty Name) -join ';'",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout.strip()
    except Exception:
        return ""


@functools.lru_cache(maxsize=1)
def detect_vendor() -> str:
    """Return ``'nvidia'``, ``'amd'`` or ``'cpu'``.

    Evidence is checked in order of reliability:
      1. ``FLASHVSR_VENDOR`` override.
      2. torch build flavour (``torch.version.hip`` => AMD, ``.cuda`` => NVIDIA).
      3. ``nvidia-smi`` on PATH.
      4. Windows video-controller name (works before torch is installed).
    """
    forced = os.environ.get("FLASHVSR_VENDOR", "").strip().lower()
    if forced in VENDORS:
        return forced

    # 1. Ask torch -- most reliable once a GPU build is installed.
    try:
        import torch  # noqa: PLC0415

        if getattr(torch.version, "hip", None):
            return "amd"
        if getattr(torch.version, "cuda", None) and torch.cuda.is_available():
            name = torch.cuda.get_device_name(0).lower()
            if any(k in name for k in ("amd", "radeon", "instinct")):
                return "amd"
            return "nvidia"
    except Exception:
        pass

    # 2. nvidia-smi is NVIDIA-only.
    if shutil.which("nvidia-smi"):
        return "nvidia"

    # 3. GPU name probe (no GPU-enabled torch installed yet).
    names = _windows_gpu_names().lower()
    if any(k in names for k in ("nvidia", "geforce", "rtx", "quadro", "tesla")):
        return "nvidia"
    if any(k in names for k in ("amd", "radeon", "instinct")):
        return "amd"
    return "cpu"


@functools.lru_cache(maxsize=4)
def _ffmpeg_has_encoder(codec: str) -> bool:
    ffmpeg = _ffmpeg_path()
    if not ffmpeg:
        return False
    try:
        out = subprocess.run(
            [ffmpeg, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout
        return codec in out
    except Exception:
        return False


def resolve_ffmpeg_codec(vendor: str | None = None) -> str:
    """ffmpeg ``-c:v`` codec, falling back to libx264 if the GPU encoder is absent."""
    vendor = vendor or detect_vendor()
    preferred, _ = _ENCODER_TABLE.get(vendor, _ENCODER_TABLE["cpu"])
    if preferred != "libx264" and not _ffmpeg_has_encoder(preferred):
        return "libx264"
    return preferred


def resolve_vhs_format(vendor: str | None = None) -> str:
    """ComfyUI VHS ``format`` string matching the resolved ffmpeg codec."""
    return _CODEC_TO_FORMAT[resolve_ffmpeg_codec(vendor)]


def ffmpeg_encoder_args(codec: str, bitrate_megabit: int = 18, crf: int = 18) -> list[str]:
    """ffmpeg output args for the trimming/re-encode pass for the given codec."""
    if codec == "h264_nvenc":
        return ["-c:v", "h264_nvenc", "-preset", "p5", "-b:v", f"{bitrate_megabit}M"]
    if codec == "h264_amf":
        return ["-c:v", "h264_amf", "-quality", "quality", "-b:v", f"{bitrate_megabit}M"]
    return ["-c:v", "libx264", "-preset", "medium", "-crf", str(crf)]


def supports_sageattention(vendor: str | None = None) -> bool:
    """SageAttention only works on NVIDIA (Triton/CUDA) and only if importable."""
    vendor = vendor or detect_vendor()
    if vendor != "nvidia":
        return False
    try:
        import sageattention  # noqa: F401,PLC0415

        return True
    except Exception:
        return False


def resolve_sageattention(preference: str = "auto", vendor: str | None = None) -> str:
    """Resolve a ``enable``/``disable``/``auto`` preference to ``enable``/``disable``.

    Never enables on hardware that cannot support it.
    """
    vendor = vendor or detect_vendor()
    if preference == "auto":
        return "enable" if supports_sageattention(vendor) else "disable"
    if preference == "enable" and not supports_sageattention(vendor):
        return "disable"
    return preference


def resolve_device(requested: str = "auto") -> str:
    """Pass-through device. ``auto`` lets the FlashVSR node pick cuda:0/mps/cpu.

    Works for ROCm too because HIP is exposed through ``torch.cuda``.
    """
    return requested


def summary(vendor: str | None = None) -> dict:
    vendor = vendor or detect_vendor()
    codec = resolve_ffmpeg_codec(vendor)
    return {
        "vendor": vendor,
        "ffmpeg_codec": codec,
        "vhs_format": _CODEC_TO_FORMAT[codec],
        "sageattention_supported": supports_sageattention(vendor),
        "device": resolve_device("auto"),
    }


if __name__ == "__main__":
    print(json.dumps(summary(), indent=2))
