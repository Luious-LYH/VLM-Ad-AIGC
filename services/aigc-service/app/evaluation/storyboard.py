"""Event-level Storyboard evaluation with visual evidence.

The evaluator combines transparent per-event visual rules with an optional
VLM review supplied by the caller.  It never labels the old frame-difference
heuristic as Storyboard adherence; that value is emitted separately as a
legacy diagnostic.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def _frame_metrics(frames: list[Image.Image], masks: list[np.ndarray | None]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    areas: list[float] = []
    centres: list[tuple[float, float]] = []
    brightness: list[float] = []
    for image, mask in zip(frames, masks):
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        gray = rgb.mean(axis=2)
        if mask is None or not np.any(mask):
            areas.append(0.0); centres.append((0.5, 0.5)); brightness.append(0.0); continue
        ys, xs = np.where(mask)
        areas.append(float(mask.mean()))
        centres.append((float(xs.mean() / max(1, image.width)), float(ys.mean() / max(1, image.height))))
        brightness.append(float(gray[mask].mean()))
    return np.asarray(areas), np.asarray(centres), np.asarray(brightness)


def _window_indices(timestamps: np.ndarray, start: float, end: float) -> list[int]:
    selected = np.where((timestamps >= start - 1e-6) & (timestamps <= end + 1e-6))[0].tolist()
    return selected or [int(np.argmin(np.abs(timestamps - (start + end) / 2)))]


def _evidence(event_id: str, indices: list[int], frames: list[Image.Image], evidence_dir: Path) -> list[str]:
    evidence_dir.mkdir(parents=True, exist_ok=True)
    chosen = [indices[0], indices[len(indices) // 2], indices[-1]] if indices else []
    paths: list[str] = []
    for rank, index in enumerate(dict.fromkeys(chosen)):
        path = evidence_dir / f"{event_id}_{rank:02d}_frame-{index:04d}.png"
        frames[index].save(path)
        paths.append(str(path))
    return paths


def evaluate_storyboard(frames: list[Image.Image], masks: list[np.ndarray | None], fps: float,
                        events: list[dict[str, Any]], evidence_dir: Path,
                        vlm_review: dict[str, Any] | None = None) -> dict[str, Any]:
    if not frames or len(frames) != len(masks):
        return {"status": "unavailable", "reason": "empty_or_mismatched_frames", "events": []}
    timestamps = np.arange(len(frames), dtype=np.float32) / max(float(fps), 1.0)
    areas, centres, brightness = _frame_metrics(frames, masks)
    event_rows: list[dict[str, Any]] = []
    previous_end = -math.inf
    for event in events:
        start, end = float(event["start_s"]), float(event["end_s"])
        indices = _window_indices(timestamps, start, end)
        event_areas = areas[indices]
        event_centres = centres[indices]
        event_brightness = brightness[indices]
        kind = str(event.get("camera_motion") or event.get("id"))
        violations: list[str] = []
        if start < previous_end:
            violations.append("event_order_overlap")
        previous_end = end
        if np.any(event_areas <= 0):
            violations.append("product_missing_in_event")
        area_ratio = float(event_areas[-1] / max(event_areas[0], 1e-6))
        centre_jitter = float(np.linalg.norm(np.diff(event_centres, axis=0), axis=1).mean()) if len(indices) > 1 else 0.0
        brightness_range = float(event_brightness.max() - event_brightness.min()) if len(indices) else 0.0
        if kind in {"macro_push_in", "push_in", "event_01"}:
            evidence = area_ratio >= 1.03
            expected = "product_scale_increases"
        elif kind in {"orbit", "slow_orbit", "event_02"}:
            horizontal_range = float(event_centres[:, 0].max() - event_centres[:, 0].min())
            evidence = horizontal_range >= 0.008 and centre_jitter <= 0.12
            expected = "continuous_horizontal_parallax"
        elif kind in {"lighting_sweep", "event_03"}:
            evidence = brightness_range >= 0.015
            expected = "foreground_brightness_changes"
        else:
            early = float(np.mean(np.abs(np.diff(areas[: max(2, len(areas) // 2)])))) if len(areas) > 2 else 0.0
            late = float(np.mean(np.abs(np.diff(areas[max(0, len(areas) // 2) :])))) if len(areas) > 2 else 0.0
            evidence = late <= max(early * 1.25, 0.02)
            expected = "late_motion_settles"
        rule_score = 0.0 if violations else (1.0 if evidence else 0.35)
        evidence_paths = _evidence(str(event.get("id", "event")), indices, frames, evidence_dir)
        vlm_row = (vlm_review or {}).get(str(event.get("id", "")))
        event_rows.append({
            "id": event.get("id"), "target_window_s": [start, end],
            "observed_window_s": [round(float(timestamps[indices[0]]), 4), round(float(timestamps[indices[-1]]), 4)],
            "completed_by_visual_rules": bool(evidence and not violations),
            "visual_evidence_score": round(rule_score, 4),
            "expected_visual_evidence": expected,
            "area_ratio": round(area_ratio, 5),
            "centre_jitter": round(centre_jitter, 5),
            "foreground_brightness_range": round(brightness_range, 5),
            "violations": violations,
            "evidence_frames": evidence_paths,
            "vlm_review": vlm_row or {"status": "unavailable", "reason": "no_storyboard_vlm_review_supplied"},
        })
    completed = [row["visual_evidence_score"] for row in event_rows]
    order_ok = not any("event_order_overlap" in row["violations"] for row in event_rows)
    continuity = 1.0 - min(1.0, sum("product_missing_in_event" in row["violations"] for row in event_rows) / max(1, len(event_rows)))
    score = (0.45 * (sum(row["completed_by_visual_rules"] for row in event_rows) / max(1, len(event_rows)))
             + 0.25 * (1.0 if order_ok else 0.0)
             + 0.20 * float(np.mean(completed) if completed else 0.0)
             + 0.10 * continuity)
    return {
        "status": "succeeded",
        "method": "event_level_visual_rules_plus_optional_vlm_review",
        "score": round(float(score), 4),
        "order_ok": order_ok,
        "continuity_score": round(float(continuity), 4),
        "events": event_rows,
        "weights": {"event_completion": 0.45, "time_and_order": 0.25, "visual_evidence": 0.20, "continuity": 0.10},
        "note": "This is an explainable project evaluator, not an official VBench score or human aesthetic rating.",
    }
