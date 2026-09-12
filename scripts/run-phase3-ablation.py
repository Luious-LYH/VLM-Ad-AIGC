"""Build the Phase 3 ablation report from completed v0.1 evaluations.

Phase 3 is intentionally offline and reproducible: it does not launch another
video model. It compares the same videos with (1) the real product mask and
the legacy centre ellipse, (2) uniform 3-frame and 16-frame sampling runs,
and (3) the legacy motion heuristic and event-level storyboard evaluator.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = PROJECT_ROOT / "runs/v0.1/sample02-fresh-provenance-v3"
DEFAULT_SAMPLE_RUN = PROJECT_ROOT / "runs/phase3/sample-02/eval-3"
DEFAULT_OUTPUT = PROJECT_ROOT / "results/phase3/sample-02"


def runtime_python() -> str:
    """Use the configured ML environment when the launcher is system Python."""
    configured = os.environ.get("AIGC_PIPELINE_PYTHON") or os.environ.get("AIGC_SERVICE_PYTHON")
    return configured if configured and Path(configured).is_file() else sys.executable


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def metric_files(run_dir: Path) -> list[Path]:
    return sorted(path for path in (run_dir / "metrics").glob("*.json") if path.name != "summary.json")


def load_metrics(run_dir: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for path in metric_files(run_dir):
        row = read_json(path)
        rows[str(row.get("video_id") or path.stem)] = row
    return rows


def run_three_sample_evaluation(args: argparse.Namespace, target: Path) -> None:
    """Create an independent uniform-3 sampling run when requested."""
    target.mkdir(parents=True, exist_ok=True)
    command = [
        runtime_python(), str(PROJECT_ROOT / "scripts/run-v01-evaluation.py"),
        "--config", str(args.config.resolve()), "--input", str(args.input.resolve()),
        "--keyframe-root", str(args.keyframe_root.resolve()), "--video-root", str(args.video_root.resolve()),
        "--record-root", str(args.record_root.resolve()), "--output-root", str(target.parent),
        "--run-id", target.name, "--max-video-samples", "3",
    ]
    if args.dino_path is not None:
        command += ["--dino-path", str(args.dino_path.resolve())]
    if args.clip_path is not None:
        command += ["--clip-path", str(args.clip_path.resolve())]
    print("+", " ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def make_report(full_rows: dict[str, dict[str, Any]], sample_rows: dict[str, dict[str, Any]] | None,
                full_run: Path, sample_run: Path | None) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for video_id, item in full_rows.items():
        product = item.get("product_consistency", {})
        real = product.get("video", {})
        ellipse = product.get("legacy_ellipse", {})
        temporal = item.get("temporal", {})
        event_storyboard = item.get("storyboard", {})
        # The curated v0.1 run includes the sample id in each video id while
        # the evaluator's generic sample run does not. Accept both forms.
        sample_item = (sample_rows or {}).get(video_id, {})
        if not sample_item and sample_rows:
            generic_id = "__".join(video_id.split("__")[:2])
            sample_item = sample_rows.get(generic_id, {})
        sample_product = sample_item.get("product_consistency", {}).get("video", {})
        rows.append({
            "video_id": video_id,
            "mask_ablation": {
                "real_product_mask": real,
                "legacy_center_ellipse": ellipse,
                "mean_delta_real_minus_ellipse": round(float(real["mean"]) - float(ellipse["mean"]), 6)
                if isinstance(real.get("mean"), (int, float)) and isinstance(ellipse.get("mean"), (int, float)) else None,
                "interpretation": "reference-preservation proxy; ellipse is diagnostic only",
            },
            "sampling_ablation": {
                "uniform_16": real,
                "uniform_3": sample_product if sample_product else {"status": "unavailable"},
                "three_frame_run": str(sample_run) if sample_item else None,
                "interpretation": "3-frame result is an independent evaluator run when present; it is less sensitive to intermittent failures than 16 uniform samples",
            },
            "storyboard_ablation": {
                "event_level_score": event_storyboard.get("score"),
                "legacy_motion_heuristic": temporal.get("legacy_motion_heuristic"),
                "interpretation": "event-level score checks configured time windows and product-region evidence; legacy value is retained for comparison",
            },
        })
    return {
        "schema_version": "metacut.phase3_ablation/v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sample_id": "perfume_flacon",
        "full_evaluation_run": str(full_run.resolve()),
        "three_sample_evaluation_run": str(sample_run.resolve()) if sample_run else None,
        "evaluation_type": "self_supervised_proxy_and_manual_storyboard_proxy",
        "rows": rows,
        "limitations": [
            "DINO measures reference-preservation similarity, not overall visual quality.",
            "The centre ellipse is included only to show why a real product mask matters; it is not the primary score.",
            "Three versus sixteen frames changes sampling coverage, not the generated video.",
            "Storyboard values are explainable project heuristics and require human side-by-side review.",
            "Prompt-original versus VLM-rewrite is not run in this v0.1 artifact because VLM does not alter the generation request by default.",
        ],
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Phase 3 消融报告", "",
        "本报告复用已完成的 v0.1 视频，不重新生成媒体。所有数值都是商品一致性/时序/分镜的可解释代理指标，不能替代人工审片。", "",
        "| 视频组合 | 真实 mask DINO mean | 中心椭圆 DINO mean | 真实-mask差值 | 16帧 DINO mean | 3帧 DINO mean | 事件级 Storyboard | 旧稳定性启发式 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["rows"]:
        mask = row["mask_ablation"]
        sampling = row["sampling_ablation"]
        storyboard = row["storyboard_ablation"]
        real = mask.get("real_product_mask", {})
        ellipse = mask.get("legacy_center_ellipse", {})
        three = sampling.get("uniform_3", {})
        legacy_stability = storyboard.get("legacy_motion_heuristic", {})
        if isinstance(legacy_stability, dict):
            legacy_stability = legacy_stability.get("temporal_stability", "—")
        lines.append(
            f"| `{row['video_id']}` | {real.get('mean', '—')} | {ellipse.get('mean', '—')} | "
            f"{mask.get('mean_delta_real_minus_ellipse', '—')} | {sampling.get('uniform_16', {}).get('mean', '—')} | "
            f"{three.get('mean', '—')} | {storyboard.get('event_level_score', '—')} | {legacy_stability} |"
        )
    lines += [
        "", "## 解释", "",
        "- `真实 mask DINO` 使用输入商品与视频商品区域的 DINOv2 embedding 余弦相似度；中心椭圆仅作旧方法对照。",
        "- `3帧/16帧` 对同一 MP4 使用不同的均匀采样密度；必须结合最低分、p10 和人工证据帧阅读。",
        "- `事件级 Storyboard` 检查时间窗口、动作证据、顺序和商品是否缺失；`旧运动启发式` 只反映帧差与时长。",
        "- 该单样本消融不支持跨数据集泛化或统计显著性结论。", "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN, help="Completed v0.1 evaluation with 16 uniform samples")
    parser.add_argument("--three-sample-run", type=Path, default=DEFAULT_SAMPLE_RUN, help="Existing independent --max-video-samples 3 run")
    parser.add_argument("--run-three-sample", action="store_true", help="Run the evaluator at 3 samples before building the report")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/phase2-comparison.json")
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "samples/sample-02/input.png")
    parser.add_argument("--keyframe-root", type=Path, default=PROJECT_ROOT / "samples/sample-02/keyframes")
    parser.add_argument("--video-root", type=Path, default=PROJECT_ROOT / "samples/sample-02/videos")
    parser.add_argument("--record-root", type=Path, default=PROJECT_ROOT / "samples/sample-02/records")
    parser.add_argument("--dino-path", type=Path, default=None)
    parser.add_argument("--clip-path", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    # Compatibility alias for older commands; Phase 3 no longer launches Hunyuan.
    parser.add_argument("--evaluate-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    full_run = args.run_dir.resolve()
    if not full_run.is_dir():
        raise FileNotFoundError(f"full evaluation run not found: {full_run}")
    sample_run = args.three_sample_run.resolve()
    if args.run_three_sample:
        run_three_sample_evaluation(args, sample_run)
    if not sample_run.is_dir() or not metric_files(sample_run):
        sample_run = None
    full_rows = load_metrics(full_run)
    if not full_rows:
        raise FileNotFoundError(f"no metric JSON files under {full_run / 'metrics'}")
    report = make_report(full_rows, load_metrics(sample_run) if sample_run else None, full_run, sample_run)
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_json(args.output_root / "ablation.json", report)
    write_markdown(report, args.output_root / "ablation.md")
    print(json.dumps({"status": "succeeded", "rows": len(report["rows"]), "output": str(args.output_root / "ablation.json"), "three_sample_run": str(sample_run) if sample_run else None}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
