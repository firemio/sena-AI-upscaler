import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hardware  # noqa: E402  (local module: scripts/hardware.py)

FFMPEG = shutil.which("ffmpeg") or r"C:\Users\starc\AppData\Local\Microsoft\WinGet\Links\ffmpeg.exe"
FFPROBE = shutil.which("ffprobe") or r"C:\Users\starc\AppData\Local\Microsoft\WinGet\Links\ffprobe.exe"

PROFILES = {
    "fast": {
        "model_version": "Tiny (Fast)",
        "tile_size": 384,
        "tile_overlap": 32,
        "speed_optimization": 1.5,
        "quality_boost": 3.0,
        "stability_level": 9,
        "color_fix": True,
        "vae_tiling": True,
        "sageattention": "disable",
        "device": "auto",
        "precision": "bf16",
        "seed": 1,
    },
    "balanced": {
        "model_version": "Full (Best Quality)",
        "tile_size": 384,
        "tile_overlap": 32,
        "speed_optimization": 1.5,
        "quality_boost": 3.0,
        "stability_level": 9,
        "color_fix": True,
        "vae_tiling": True,
        "sageattention": "enable",
        "device": "auto",
        "precision": "bf16",
        "seed": 1,
    },
    "degraded_action": {
        "model_version": "Full (Best Quality)",
        "tile_size": 384,
        "tile_overlap": 32,
        "speed_optimization": 1.5,
        "quality_boost": 3.0,
        "stability_level": 9,
        "color_fix": True,
        "vae_tiling": True,
        "sageattention": "enable",
        "device": "auto",
        "precision": "bf16",
        "seed": 1,
    },
}


def run(cmd, cwd=None):
    return subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)


def probe(path: Path):
    raw = run(
        [
            FFPROBE,
            "-v",
            "error",
            "-show_entries",
            "stream=avg_frame_rate",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ]
    ).stdout
    data = json.loads(raw)
    fps_text = data["streams"][0]["avg_frame_rate"]
    num, den = fps_text.split("/")
    fps = float(num) / float(den)
    duration = float(data["format"]["duration"])
    return fps, duration


def probe_dimensions(path: Path):
    raw = run(
        [
            FFPROBE,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
            str(path),
        ]
    ).stdout
    data = json.loads(raw)
    stream = data["streams"][0]
    return int(stream["width"]), int(stream["height"])


def detect_scene_cuts(
    path: Path,
    total_frames: int,
    threshold: float,
    min_gap_frames: int,
    analysis_width: int = 160,
):
    if threshold <= 0:
        return []

    width, height = probe_dimensions(path)
    scaled_height = max(8, int(round(height * (analysis_width / width))))
    scaled_height += scaled_height % 2
    frame_size = analysis_width * scaled_height

    cmd = [
        FFMPEG,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-vf",
        f"scale={analysis_width}:{scaled_height}:flags=bilinear,format=gray",
        "-vsync",
        "0",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-",
    ]
    proc = subprocess.run(cmd, check=True, capture_output=True)
    raw = proc.stdout
    if len(raw) < frame_size * 2:
        return []

    frames = np.frombuffer(raw, dtype=np.uint8)
    frames = frames[: (len(frames) // frame_size) * frame_size]
    if frames.size < frame_size * 2:
        return []

    frames = frames.reshape(-1, frame_size).astype(np.float32)
    diffs = np.mean(np.abs(frames[1:] - frames[:-1]), axis=1) / 255.0

    cut_frames = []
    last_cut = -min_gap_frames
    for idx, score in enumerate(diffs, start=1):
        if score < threshold:
            continue
        if idx - last_cut < min_gap_frames:
            continue
        if idx <= 0 or idx >= total_frames:
            continue
        cut_frames.append(idx)
        last_cut = idx
    return cut_frames


def plan_chunks(total_frames: int, chunk_frames: int, overlap_frames: int, cut_frames, min_chunk_frames: int = 21):
    chunks = []
    start_frame = 0
    drop_frames = 0
    cut_frames = sorted(cut_frames)

    while start_frame < total_frames:
        remaining = total_frames - start_frame
        if remaining < min_chunk_frames and chunks:
            chunks[-1]["frame_cap"] += remaining
            break

        if remaining <= chunk_frames:
            chunks.append(
                {
                    "start_frame": start_frame,
                    "frame_cap": remaining,
                    "drop_frames": drop_frames,
                }
            )
            break

        max_end = start_frame + chunk_frames
        scene_end = None
        for cut in cut_frames:
            if cut < start_frame + min_chunk_frames:
                continue
            if cut > max_end:
                break
            if total_frames - cut < min_chunk_frames:
                continue
            scene_end = cut
            break

        if scene_end is not None:
            frame_cap = scene_end - start_frame
            chunks.append(
                {
                    "start_frame": start_frame,
                    "frame_cap": frame_cap,
                    "drop_frames": drop_frames,
                }
            )
            start_frame = scene_end
            drop_frames = 0
            continue

        chunks.append(
            {
                "start_frame": start_frame,
                "frame_cap": chunk_frames,
                "drop_frames": drop_frames,
            }
        )
        start_frame += chunk_frames - overlap_frames
        drop_frames = overlap_frames

    return chunks


def trim_chunk(src: Path, dst: Path, drop_frames: int, fps: float, codec: str):
    if drop_frames <= 0:
        shutil.copy2(src, dst)
        return
    vf = f"select='gte(n\\,{drop_frames})',setpts=N/({fps}*TB)"
    run(
        [
            FFMPEG,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(src),
            "-vf",
            vf,
            "-an",
            *hardware.ffmpeg_encoder_args(codec),
            str(dst),
        ]
    )


def concat_chunks(chunk_paths, original_video: Path, final_path: Path, list_path: Path):
    list_path.write_text(
        "".join(f"file '{p.as_posix()}'\n" for p in chunk_paths),
        encoding="utf-8",
    )
    video_only = final_path.with_name(final_path.stem + ".video_only.mp4")
    run(
        [
            FFMPEG,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
            "-c",
            "copy",
            str(video_only),
        ]
    )
    run(
        [
            FFMPEG,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_only),
            "-i",
            str(original_video),
            "-map",
            "0:v:0",
            "-map",
            "1:a?",
            "-c:v",
            "copy",
            "-c:a",
            "copy",
            "-shortest",
            str(final_path),
        ]
    )


def run_job(
    video_path: Path,
    prefix: str,
    start_time: float,
    frame_load_cap: int,
    profile: dict,
    output_format: str,
    sageattention: str,
    device: str,
):
    cmd = [
        sys.executable,
        str(Path(__file__).with_name("run_flashvsr_job.py")),
        str(video_path),
        prefix,
        "--start-time",
        str(start_time),
        "--frame-load-cap",
        str(frame_load_cap),
        "--model-version",
        profile["model_version"],
        "--output-format",
        output_format,
        "--tile-size",
        str(profile["tile_size"]),
        "--tile-overlap",
        str(profile["tile_overlap"]),
        "--speed-optimization",
        str(profile["speed_optimization"]),
        "--quality-boost",
        str(profile["quality_boost"]),
        "--stability-level",
        str(profile["stability_level"]),
        "--sageattention",
        sageattention,
        "--device",
        device,
        "--precision",
        str(profile["precision"]),
        "--seed",
        str(profile["seed"]),
    ]
    if profile["color_fix"]:
        cmd.append("--color-fix")
    else:
        cmd.append("--no-color-fix")
    if profile["vae_tiling"]:
        cmd.append("--vae-tiling")
    else:
        cmd.append("--no-vae-tiling")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout + "\n" + proc.stderr)

    output_path = None
    for line in proc.stdout.splitlines():
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        outputs = obj.get("outputs", {})
        if "4" in outputs:
            gifs = outputs["4"].get("gifs", [])
            if gifs:
                output_path = Path(gifs[0]["fullpath"])
    if not output_path:
        raise RuntimeError(proc.stdout)
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path")
    parser.add_argument("--chunk-frames", type=int, default=144)
    parser.add_argument("--overlap-frames", type=int, default=32)
    parser.add_argument("--scene-cut-threshold", type=float, default=0.24)
    parser.add_argument("--scene-cut-min-gap", type=int, default=18)
    parser.add_argument("--run-name", default="flashvsr_full_run")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="balanced")
    parser.add_argument(
        "--encoder",
        choices=["auto", "nvenc", "amf", "libx264"],
        default="auto",
        help="Video encoder. 'auto' picks NVENC on NVIDIA, AMF on AMD, libx264 otherwise.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device passed to FlashVSR ('auto' resolves to cuda:0/mps/cpu; ROCm uses cuda).",
    )
    parser.add_argument(
        "--sageattention",
        choices=["auto", "enable", "disable"],
        default="auto",
        help="'auto' enables SageAttention only on NVIDIA when the package is installed.",
    )
    args = parser.parse_args()

    video_path = Path(args.video_path).resolve()
    profile = PROFILES[args.profile]

    # Resolve the hardware-dependent knobs once so the same run works on
    # NVIDIA (CUDA) and AMD (ROCm) without editing the profiles.
    vendor = hardware.detect_vendor()
    if args.encoder == "auto":
        trim_codec = hardware.resolve_ffmpeg_codec(vendor)
        vhs_format = hardware.resolve_vhs_format(vendor)
    else:
        trim_codec = {"nvenc": "h264_nvenc", "amf": "h264_amf", "libx264": "libx264"}[args.encoder]
        vhs_format = {
            "nvenc": "video/nvenc_h264-mp4",
            "amf": "video/amf_h264-mp4",
            "libx264": "video/h264-mp4",
        }[args.encoder]
    device = args.device if args.device != "auto" else profile["device"]
    sageattention = (
        hardware.resolve_sageattention(profile["sageattention"], vendor)
        if args.sageattention == "auto"
        else hardware.resolve_sageattention(args.sageattention, vendor)
    )
    print(
        json.dumps(
            {
                "hardware": {
                    "vendor": vendor,
                    "encoder": trim_codec,
                    "vhs_format": vhs_format,
                    "device": device,
                    "sageattention": sageattention,
                }
            },
            ensure_ascii=True,
        ),
        flush=True,
    )

    fps, duration = probe(video_path)
    total_frames = math.ceil(duration * fps)
    cut_frames = detect_scene_cuts(
        video_path,
        total_frames=total_frames,
        threshold=args.scene_cut_threshold,
        min_gap_frames=args.scene_cut_min_gap,
    )
    chunk_plan = plan_chunks(
        total_frames=total_frames,
        chunk_frames=args.chunk_frames,
        overlap_frames=args.overlap_frames,
        cut_frames=cut_frames,
    )

    base_dir = Path("runs") / args.run_name
    raw_dir = base_dir / "raw_chunks"
    trim_dir = base_dir / "trimmed_chunks"
    base_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    trim_dir.mkdir(parents=True, exist_ok=True)
    log_path = base_dir / "progress.jsonl"
    plan_path = base_dir / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "fps": fps,
                "duration": duration,
                "total_frames": total_frames,
                "scene_cut_threshold": args.scene_cut_threshold,
                "scene_cut_min_gap": args.scene_cut_min_gap,
                "profile": args.profile,
                "profile_settings": profile,
                "hardware": {
                    "vendor": vendor,
                    "encoder": trim_codec,
                    "vhs_format": vhs_format,
                    "device": device,
                    "sageattention": sageattention,
                },
                "scene_cuts": cut_frames,
                "chunks": chunk_plan,
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )

    trimmed_paths = []
    for idx, chunk in enumerate(chunk_plan):
        start_frame = chunk["start_frame"]
        frame_cap = chunk["frame_cap"]
        drop_frames = chunk["drop_frames"]
        if frame_cap < 21:
            continue
        start_time = start_frame / fps
        prefix = f"{args.run_name}_chunk_{idx:04d}"

        raw_output = raw_dir / f"{prefix}.mp4"
        trimmed_output = trim_dir / f"{prefix}.mp4"

        if not raw_output.exists():
            generated = run_job(
                video_path,
                prefix,
                start_time,
                frame_cap,
                profile,
                vhs_format,
                sageattention,
                device,
            )
            shutil.move(str(generated), raw_output)

        if not trimmed_output.exists():
            trim_chunk(raw_output, trimmed_output, drop_frames, fps, trim_codec)

        trimmed_paths.append(trimmed_output.resolve())
        with log_path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "chunk": idx,
                        "start_frame": start_frame,
                        "start_time": start_time,
                        "frame_cap": frame_cap,
                        "drop_frames": drop_frames,
                        "raw": str(raw_output),
                        "trimmed": str(trimmed_output),
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )

    final_path = base_dir / f"{args.run_name}.final.mp4"
    concat_chunks(trimmed_paths, video_path, final_path, base_dir / "concat_list.txt")
    print(json.dumps({"final": str(final_path.resolve())}, ensure_ascii=True))


if __name__ == "__main__":
    main()
