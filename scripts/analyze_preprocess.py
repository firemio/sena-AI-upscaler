import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


def run(cmd):
    return subprocess.run(cmd, check=True, capture_output=True)


def probe_video(path: str) -> dict:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        path,
    ]
    data = json.loads(run(cmd).stdout.decode("utf-8"))
    stream = next(s for s in data["streams"] if s.get("codec_type") == "video")
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "field_order": stream.get("field_order", "unknown"),
        "r_frame_rate": stream.get("r_frame_rate", "0/0"),
        "avg_frame_rate": stream.get("avg_frame_rate", "0/0"),
        "pix_fmt": stream.get("pix_fmt", ""),
        "duration": float(stream.get("duration") or data.get("format", {}).get("duration") or 0.0),
    }


def parse_rate(rate: str) -> float:
    if not rate or rate == "0/0":
        return 0.0
    num, den = rate.split("/")
    num = float(num)
    den = float(den)
    return 0.0 if den == 0 else num / den


def is_interlaced(field_order: str) -> bool:
    value = (field_order or "").lower()
    return value not in {"", "unknown", "progressive"}


def sample_detail_score(path: str, sample_count: int = 24, width: int = 320) -> float:
    probe = probe_video(path)
    src_w = probe["width"]
    src_h = probe["height"]
    duration = max(probe["duration"], 1.0)
    fps = max(parse_rate(probe["avg_frame_rate"]), 1.0)
    interval = max(duration / sample_count, 0.5)
    out_h = max(2, int(round(src_h * (width / src_w))))

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        path,
        "-vf",
        f"fps=1/{interval:.6f},scale={width}:{out_h}:flags=lanczos,format=gray",
        "-frames:v",
        str(sample_count),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-",
    ]
    raw = run(cmd).stdout
    frame_size = width * out_h
    if len(raw) < frame_size:
        return 0.0

    scores = []
    for offset in range(0, len(raw) - frame_size + 1, frame_size):
        frame = np.frombuffer(raw[offset : offset + frame_size], dtype=np.uint8).reshape(out_h, width).astype(np.float32)
        center = frame[1:-1, 1:-1]
        lap = np.abs(
            4.0 * center
            - frame[:-2, 1:-1]
            - frame[2:, 1:-1]
            - frame[1:-1, :-2]
            - frame[1:-1, 2:]
        )
        scores.append(float(lap.mean()))
    return float(np.mean(scores)) if scores else 0.0


def sample_motion_score(path: str, sample_count: int = 48, width: int = 160) -> float:
    probe = probe_video(path)
    src_w = probe["width"]
    src_h = probe["height"]
    duration = max(probe["duration"], 1.0)
    interval = max(duration / sample_count, 0.25)
    out_h = max(2, int(round(src_h * (width / src_w))))

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        path,
        "-vf",
        f"fps=1/{interval:.6f},scale={width}:{out_h}:flags=bilinear,format=gray",
        "-frames:v",
        str(sample_count),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-",
    ]
    raw = run(cmd).stdout
    frame_size = width * out_h
    if len(raw) < frame_size * 2:
        return 0.0

    frames = np.frombuffer(raw, dtype=np.uint8)
    frames = frames[: (len(frames) // frame_size) * frame_size]
    if frames.size < frame_size * 2:
        return 0.0

    frames = frames.reshape(-1, frame_size).astype(np.float32)
    diffs = np.mean(np.abs(frames[1:] - frames[:-1]), axis=1) / 255.0
    return float(np.mean(diffs)) if len(diffs) else 0.0


def recommend_profile(detail_score: float, motion_score: float) -> str:
    if detail_score <= 14.0:
        return "degraded_action"
    if detail_score <= 18.0:
        return "balanced"
    return "fast"


def preprocess(input_path: str, output_path: str, pseudo_fhd: bool, interlaced: bool):
    filters = []
    if interlaced:
        filters.append("yadif=mode=send_frame:parity=auto:deint=interlaced")
    if pseudo_fhd:
        filters.append("scale=720:480:flags=lanczos")

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        input_path,
    ]
    if filters:
        cmd += ["-vf", ",".join(filters)]
    cmd += [
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c:v",
        "ffv1",
        "-level",
        "3",
        "-g",
        "1",
        "-c:a",
        "copy",
        output_path,
    ]
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--detail-threshold", type=float, default=12.0)
    args = parser.parse_args()

    input_path = str(Path(args.input).resolve())
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    meta = probe_video(input_path)
    detail_score = sample_detail_score(input_path)
    motion_score = sample_motion_score(input_path)
    pseudo_fhd = meta["width"] >= 1920 and meta["height"] >= 1080 and detail_score < args.detail_threshold
    interlaced = is_interlaced(meta["field_order"])
    profile = recommend_profile(detail_score, motion_score)

    stem = Path(input_path).stem
    normalized = output_dir / f"{stem}.normalized.mkv"
    report = output_dir / f"{stem}.analysis.json"

    preprocess(input_path, str(normalized), pseudo_fhd=pseudo_fhd, interlaced=interlaced)

    result = {
        "input": input_path,
        "normalized_video": str(normalized),
        "width": meta["width"],
        "height": meta["height"],
        "field_order": meta["field_order"],
        "interlaced": interlaced,
        "detail_score": round(detail_score, 4),
        "motion_score": round(motion_score, 4),
        "pseudo_fhd": pseudo_fhd,
        "fps": round(parse_rate(meta["avg_frame_rate"]) or parse_rate(meta["r_frame_rate"]), 6),
        "recommended_profile": profile,
    }
    report.write_text(json.dumps(result, ensure_ascii=True, indent=2), encoding="utf-8")
    sys.stdout.write(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
