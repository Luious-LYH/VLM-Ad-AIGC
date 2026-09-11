"""Build a compact, resume-friendly Phase-2 comparison report.

The media pipeline writes one detailed ``record.json`` per sample and the
evaluator writes one ``metrics.json`` per sample.  This script deliberately
does not recompute any metric; it only joins those records into a JSON summary
and a Markdown table that is convenient to inspect or attach to a resume/demo.
"""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


parser = argparse.ArgumentParser()
parser.add_argument("--phase2-root", type=Path, default=Path("public/uploads/phase2"))
parser.add_argument("--run-root", type=Path, default=Path("runs/phase2"))
args = parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def mean(values: list[Any]) -> float | None:
    clean = [float(v) for v in values if number(v) is not None]
    return round(statistics.mean(clean), 4) if clean else None


def sample_row(combo_id: str, sample_dir: Path) -> dict[str, Any]:
    record = read_json(sample_dir / "record.json")
    metrics = read_json(sample_dir / "metrics.json") if (sample_dir / "metrics.json").is_file() else {}
    vlm = record.get("vlm", {})
    vlm_metrics = vlm.get("metrics", {})
    image = record.get("image", {})
    image_result = image.get("result", {})
    video = record.get("video", {})
    video_result = video.get("result", {})
    video_metrics = metrics.get("video", {})
    video_output = video.get("output")
    if not isinstance(video_output, str) or not video_output:
        video_output = str((sample_dir / "final.mp4").resolve())
    return {
        "combo_id": combo_id,
        "sample_id": record.get("sample_id"),
        "status": record.get("status", "failed"),
        "vlm_model": vlm.get("model"),
        "image_model": image.get("model"),
        "video_model": video.get("model"),
        "json_schema_pass": vlm_metrics.get("json_schema_pass"),
        "attribute_recognition": number(vlm_metrics.get("attribute_recognition")),
        "ocr_accuracy": number(vlm_metrics.get("ocr_accuracy")),
        "vlm_latency_ms": number(vlm.get("latency_ms")),
        "vlm_peak_vram_mb": number(vlm.get("peak_vram_mb")),
        "image_latency_ms": number(image_result.get("latency_ms")),
        "image_peak_vram_mb": number(image_result.get("peak_vram_mb")),
        "video_latency_ms": number(video_result.get("latency_ms")),
        "video_peak_vram_mb": number(video_result.get("peak_vram_mb")),
        "masked_dino_product_similarity": number(metrics.get("masked_dino_product_similarity")),
        "masked_dino_keyframe_similarity": number(metrics.get("masked_dino_keyframe_similarity")),
        "clip_text_image_match": number(metrics.get("clip_text_image_match")),
        "clip_text_video_match": number(metrics.get("clip_text_video_match")),
        "stability": number(video_metrics.get("stability")),
        "storyboard_adherence": number(video_metrics.get("storyboard_adherence")),
        "frame_count": video_metrics.get("probe", {}).get("frame_count"),
        "fps": video_metrics.get("probe", {}).get("fps"),
        "width": video_metrics.get("probe", {}).get("width"),
        "height": video_metrics.get("probe", {}).get("height"),
        "video_path": str(Path(video_output).as_posix()),
    }


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = [
        "attribute_recognition",
        "ocr_accuracy",
        "vlm_latency_ms",
        "vlm_peak_vram_mb",
        "image_latency_ms",
        "image_peak_vram_mb",
        "video_latency_ms",
        "video_peak_vram_mb",
        "masked_dino_product_similarity",
        "masked_dino_keyframe_similarity",
        "clip_text_image_match",
        "clip_text_video_match",
        "stability",
        "storyboard_adherence",
    ]
    result: list[dict[str, Any]] = []
    for combo_id in sorted({row["combo_id"] for row in rows}):
        group = [row for row in rows if row["combo_id"] == combo_id]
        first = group[0]
        result.append(
            {
                "combo_id": combo_id,
                "vlm_model": first["vlm_model"],
                "image_model": first["image_model"],
                "video_model": first["video_model"],
                "samples": len(group),
                "success_rate": round(sum(row["status"] == "succeeded" for row in group) / len(group), 4),
                "schema_pass_rate": round(sum(row["json_schema_pass"] is True for row in group) / len(group), 4),
                **{key: mean([row[key] for row in group]) for key in keys},
            }
        )
    return result


def fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    rows: list[dict[str, Any]] = []
    for combo_dir in sorted(args.phase2_root.iterdir()):
        if not combo_dir.is_dir() or combo_dir.name.startswith("_"):
            continue
        for sample_dir in sorted(combo_dir.glob("sample-*")):
            if (sample_dir / "record.json").is_file():
                rows.append(sample_row(combo_dir.name, sample_dir))
    summary = {
        "schema_version": "metacut.phase2_report/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sample_count": len(rows),
        "combination_count": len({row["combo_id"] for row in rows}),
        "rows": rows,
        "combinations": aggregate(rows),
    }
    args.run_root.mkdir(parents=True, exist_ok=True)
    (args.run_root / "phase2-report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# MetaCut Phase-2/2.5 controlled model comparison",
        "",
        f"Generated: `{summary['generated_at']}` · samples: **{len(rows)}** · combinations: **{summary['combination_count']}**",
        "",
        "All combinations use the same input image family, prompt, seed and output specification. Metrics marked `—` are unavailable, not fabricated.",
        "",
        "| Combination | Samples | Success | Schema | Attr | OCR | DINO | CLIP-I | CLIP-V | Stability | Storyboard | VLM ms | Video ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary["combinations"]:
        lines.append(
            "| {combo_id} | {samples} | {success_rate} | {schema_pass_rate} | {attribute_recognition} | {ocr_accuracy} | {masked_dino_product_similarity} | {clip_text_image_match} | {clip_text_video_match} | {stability} | {storyboard_adherence} | {vlm_latency_ms} | {video_latency_ms} |".format(
                **{key: fmt(value) for key, value in item.items()}
            )
        )
    lines += ["", "## Sample-level artifacts", ""]
    for row in rows:
        lines.append(f"- `{row['combo_id']}/{row['sample_id']}` → `{row['video_path']}`")
    (args.run_root / "phase2-report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.run_root / "phase2-report.json")
    print(args.run_root / "phase2-report.md")


if __name__ == "__main__":
    main()
