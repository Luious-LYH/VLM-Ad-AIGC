"""Re-encode Phase-2 previews for presentation quality without changing media specs."""

from __future__ import annotations

import argparse
import os
from fractions import Fraction
from pathlib import Path

import av


parser = argparse.ArgumentParser()
parser.add_argument("--phase2-root", type=Path, default=Path("public/uploads/phase2"))
parser.add_argument("--shared-video-root", type=Path, default=None)
parser.add_argument("--crf", type=int, default=17)
parser.add_argument("--preset", default="medium")
args = parser.parse_args()


def transcode(source_path: Path, output_path: Path) -> tuple[int, float, int]:
    temp = output_path.with_suffix(".hq.tmp.mp4")
    frame_count = 0
    with av.open(str(source_path)) as source, av.open(str(temp), "w", format="mp4") as target:
        source_stream = source.streams.video[0]
        rate = source_stream.average_rate or 24
        output_stream = target.add_stream("libx264", rate=rate)
        output_stream.width = source_stream.width
        output_stream.height = source_stream.height
        output_stream.pix_fmt = "yuv420p"
        output_stream.options = {"crf": str(args.crf), "preset": args.preset, "movflags": "+faststart"}
        for frame in source.decode(video=0):
            frame.pts = frame_count
            frame.time_base = Fraction(1, int(rate))
            for packet in output_stream.encode(frame):
                target.mux(packet)
            frame_count += 1
        for packet in output_stream.encode():
            target.mux(packet)
    size = temp.stat().st_size
    os.replace(temp, output_path)
    return frame_count, float(rate), size


def main() -> None:
    shared_root = args.shared_video_root or args.phase2_root / "_shared" / "videos"
    videos = sorted(args.phase2_root.glob("*/sample-*/final.mp4"))
    results = []
    for path in videos:
        record_path = path.parent / "record.json"
        source_path = path
        if shared_root.is_dir() and record_path.is_file():
            import json

            record = json.loads(record_path.read_text(encoding="utf-8"))
            original_name = Path(str(record.get("video", {}).get("output", ""))).name
            candidate = shared_root / original_name
            if candidate.is_file():
                source_path = candidate
        results.append((path, *transcode(source_path, path)))
        print(f"{path}: frames={results[-1][1]} fps={results[-1][2]} bytes={results[-1][3]}", flush=True)
    print(f"transcoded={len(results)}")


if __name__ == "__main__":
    main()
