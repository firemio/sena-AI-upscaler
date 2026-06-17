import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


SERVER = os.environ.get("COMFYUI_SERVER", "http://127.0.0.1:8188")


def post_json(path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        SERVER + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_json(path: str, timeout: int = 120) -> dict:
    with urllib.request.urlopen(SERVER + path, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def build_prompt(
    video_path: str,
    prefix: str,
    start_time: float,
    frame_load_cap: int,
    model_version: str,
    output_format: str,
    include_audio: bool,
    tile_size: int,
    tile_overlap: int,
    speed_optimization: float,
    quality_boost: float,
    stability_level: int,
    color_fix: bool,
    vae_tiling: bool,
    sageattention: str,
    device: str,
    precision: str,
    seed: int,
) -> dict:
    upscale_inputs = {
        "frames": ["1", 0],
        "model_version": model_version,
        "scale": 2,
        "enable_tiling": True,
        "tile_size": tile_size,
        "tile_overlap": tile_overlap,
        "speed_optimization": speed_optimization,
        "quality_boost": quality_boost,
        "stability_level": stability_level,
        "color_fix": color_fix,
        "vae_tiling": vae_tiling,
        "unload_model": False,
        "sageattention": sageattention,
        "device": device,
        "precision": precision,
        "seed": seed,
    }
    combine_inputs = {
        "images": ["3", 0],
        "frame_rate": ["2", 0],
        "loop_count": 0,
        "filename_prefix": prefix,
        "format": output_format,
        "save_metadata": False,
        "pingpong": False,
        "save_output": True,
    }
    # GPU hardware encoders (NVIDIA NVENC / AMD AMF) share the bitrate/megabit knobs.
    if output_format in (
        "video/nvenc_h264-mp4",
        "video/nvenc_hevc-mp4",
        "video/amf_h264-mp4",
        "video/amf_hevc-mp4",
    ):
        combine_inputs.update({"pix_fmt": "yuv420p", "bitrate": 18, "megabit": True})
    elif output_format in ("video/h264-mp4", "video/h265-mp4"):
        # CPU software encoders (libx264 / libx265) are quality-driven via CRF.
        combine_inputs.update({"pix_fmt": "yuv420p", "crf": 18})
    elif output_format == "video/ffv1-mkv":
        combine_inputs.update(
            {
                "level": "3",
                "coder": "1",
                "context": "1",
                "gop_size": 1,
                "slices": "16",
                "slicecrc": "1",
                "pix_fmt": "yuv420p",
                "trim_to_audio": False,
            }
        )
    if include_audio:
        upscale_inputs["audio"] = ["1", 2]
        combine_inputs["audio"] = ["3", 1]

    return {
        "1": {
            "class_type": "VHS_LoadVideoFFmpegPath",
            "inputs": {
                "video": video_path,
                "force_rate": 0,
                "custom_width": 0,
                "custom_height": 0,
                "frame_load_cap": frame_load_cap,
                "start_time": start_time,
                "format": "None",
            },
        },
        "2": {
            "class_type": "VHS_VideoInfoSource",
            "inputs": {
                "video_info": ["1", 3],
            },
        },
        "3": {
            "class_type": "AILab_FlashVSR_Advanced",
            "inputs": upscale_inputs,
        },
        "4": {
            "class_type": "VHS_VideoCombine",
            "inputs": combine_inputs,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path")
    parser.add_argument("filename_prefix")
    parser.add_argument("--start-time", type=float, default=0.0)
    parser.add_argument("--frame-load-cap", type=int, default=0)
    parser.add_argument("--model-version", default="Tiny (Fast)")
    parser.add_argument("--output-format", default="video/nvenc_h264-mp4")
    parser.add_argument("--include-audio", action="store_true")
    parser.add_argument("--tile-size", type=int, default=384)
    parser.add_argument("--tile-overlap", type=int, default=32)
    parser.add_argument("--speed-optimization", type=float, default=1.5)
    parser.add_argument("--quality-boost", type=float, default=3.0)
    parser.add_argument("--stability-level", type=int, default=9)
    parser.add_argument("--color-fix", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vae-tiling", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sageattention", default="disable")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", default="bf16")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    video_path = str(Path(args.video_path).resolve())
    prefix = args.filename_prefix
    prompt = build_prompt(
        video_path=video_path,
        prefix=prefix,
        start_time=args.start_time,
        frame_load_cap=args.frame_load_cap,
        model_version=args.model_version,
        output_format=args.output_format,
        include_audio=args.include_audio,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
        speed_optimization=args.speed_optimization,
        quality_boost=args.quality_boost,
        stability_level=args.stability_level,
        color_fix=args.color_fix,
        vae_tiling=args.vae_tiling,
        sageattention=args.sageattention,
        device=args.device,
        precision=args.precision,
        seed=args.seed,
    )

    queued = post_json("/prompt", {"prompt": prompt})
    prompt_id = queued["prompt_id"]
    print(json.dumps({"queued": True, "prompt_id": prompt_id}, ensure_ascii=True), flush=True)

    while True:
        try:
            history = get_json(f"/history/{prompt_id}", timeout=30)
        except TimeoutError:
            print(json.dumps({"waiting": True, "prompt_id": prompt_id, "history_timeout": True}, ensure_ascii=True), flush=True)
            time.sleep(15)
            continue
        except urllib.error.URLError:
            print(json.dumps({"waiting": True, "prompt_id": prompt_id, "server_unreachable": True}, ensure_ascii=True), flush=True)
            time.sleep(15)
            continue
        if prompt_id in history:
            item = history[prompt_id]
            status = item.get("status", {})
            if status.get("status_str") == "error":
                print(json.dumps(item, ensure_ascii=True), flush=True)
                raise SystemExit(1)
            outputs = item.get("outputs", {})
            print(json.dumps({"completed": True, "prompt_id": prompt_id, "outputs": outputs}, ensure_ascii=True), flush=True)
            return
        queue = get_json("/queue")
        running = queue.get("queue_running", [])
        pending = queue.get("queue_pending", [])
        print(json.dumps({"waiting": True, "prompt_id": prompt_id, "running": len(running), "pending": len(pending)}, ensure_ascii=True), flush=True)
        time.sleep(15)


if __name__ == "__main__":
    main()
