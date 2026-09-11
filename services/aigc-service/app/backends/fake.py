from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from ..config import Settings
from ..schemas import RunMetadata
from .base import AigcBackend


def _digest(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


class FakeBackend(AigcBackend):
    """Deterministic contract backend used for UI integration and CI.

    It has no torch/transformers/diffusers imports. Inputs containing the exact
    token ``__fake_fail__`` exercise the asynchronous failed state in tests.
    """

    name = "fake"

    def __init__(self, cfg: Settings):
        self.cfg = cfg

    def capabilities(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "ready": True,
            "local_only": True,
            "operations": [
                "reference.analyze", "product.parse", "material.analyze",
                "script.stitch", "product.polish", "director.intent",
                "image.generate", "video.generate",
            ],
            "models": {
                "vlm": {"id": self.cfg.vlm_model, "revision": self.cfg.vlm_revision, "loaded": False},
                "image": {"id": self.cfg.image_model, "revision": self.cfg.image_revision, "loaded": False},
                "video": {"id": self.cfg.video_model, "revision": self.cfg.video_revision, "loaded": False},
            },
        }

    def _run(self, operation: str, payload: dict[str, Any], model_id: str, revision: str, seed: int | None = None) -> RunMetadata:
        token = _digest({"operation": operation, "payload": payload})[:16]
        return RunMetadata(
            run_id=f"run_{token}", backend=self.name, model_id=model_id,
            model_revision=revision, seed=seed, latency_ms=1, deterministic=True,
        )

    def invoke(self, operation: str, payload: dict[str, Any]) -> tuple[dict[str, Any], RunMetadata]:
        run = self._run(operation, payload, self.cfg.vlm_model, self.cfg.vlm_revision)
        handlers = {
            "reference.analyze": self._reference,
            "product.parse": self._product,
            "material.analyze": self._material,
            "script.stitch": self._script,
            "product.polish": self._polish,
            "director.intent": self._director,
        }
        return handlers[operation](payload), run

    def generation_run(self, kind: str, payload: dict[str, Any]) -> RunMetadata:
        is_image = kind == "image_generation"
        return self._run(
            kind, payload,
            payload.get("model") or (self.cfg.image_model if is_image else self.cfg.video_model),
            self.cfg.image_revision if is_image else self.cfg.video_revision,
            payload.get("seed"),
        )

    def generate(self, kind: str, payload: dict[str, Any], run: RunMetadata) -> dict[str, Any]:
        if "__fake_fail__" in payload.get("prompt", ""):
            raise RuntimeError("deterministic fake generation failure")
        token = _digest({"kind": kind, "payload": payload})[:20]
        if kind == "image_generation":
            count = payload.get("num_candidates", 1)
            artifacts = [
                {"url": f"/fake-artifacts/{token}-{index}.png", "seed": payload.get("seed", 42) + index}
                for index in range(count)
            ]
            return {
                "artifacts": artifacts,
                "generated_images": [artifact["url"] for artifact in artifacts],
                "selected_image": artifacts[0]["url"],
                "candidate_scores": [
                    {"index": index, "score": round(1 - index * 0.05, 3), "backend": payload.get("model", "fake")}
                    for index in range(count)
                ],
                "model_id": payload.get("model") or self.cfg.image_model,
                "model_revision": self.cfg.image_revision,
                "seed": payload.get("seed", 42),
                "latency_ms": run.latency_ms,
                "peak_vram_mb": 0,
                "width": payload["width"], "height": payload["height"],
                "conditioning": payload.get("conditioning", {}),
            }
        video_url = f"/fake-artifacts/{token}.mp4"
        return {
            "artifacts": [{"url": video_url, "seed": payload.get("seed", 42)}],
            "video_url": video_url,
            "model_id": payload.get("model") or self.cfg.video_model,
            "model_revision": self.cfg.video_revision,
            "seed": payload.get("seed", 42),
            "latency_ms": run.latency_ms,
            "peak_vram_mb": 0,
            "duration_seconds": payload["duration_seconds"], "fps": payload["fps"],
            "width": payload["width"], "height": payload["height"],
        }

    @staticmethod
    def _reference(payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "schema_version": "video_analysis/v1", "source_uri": payload["video_url"],
            "meta_info": {"duration": 12.0, "resolution": "1080x1920", "aspect_ratio": "9:16"},
            "narrative_structure": {"primary_type": "problem_solution", "timeline_events": [
                {"start": 0.0, "end": 2.0, "event_name": "hook", "description": "产品痛点钩子"},
                {"start": 2.0, "end": 9.0, "event_name": "product_detail", "description": "商品展示与使用"},
                {"start": 9.0, "end": 12.0, "event_name": "cta", "description": "行动号召"},
            ]},
            "rhythm_and_density": {"shot_count": 3, "avg_shot_duration": 4.0, "avg_words_per_second": 2.5},
            "on_screen_texts": [],
        }

    @staticmethod
    def _product(payload: dict[str, Any]) -> dict[str, Any]:
        description = payload.get("product_description", "").strip()
        name = re.split(r"[,，。\s]", description, maxsplit=1)[0] if description else "参考图商品"
        return {
            "schema_version": "product/v1", "product_name": name[:40], "category": "其他",
            "visual_description": description or "从多模态素材解析的商品",
            "usage_method": "按商品说明使用", "core_selling_points": ["外观辨识度", "使用便捷"],
            "target_audience": "目标消费者", "usage_scene": "日常场景",
            "pain_point_gain_point": "以直观演示呈现商品价值",
            "ingredients_material": None, "spec_size": None,
            "user_asset_inventory": {
                "product_images_count": len(payload.get("product_image_urls", [])),
                "product_video_clips_count": len(payload.get("product_video_urls", [])),
                "has_logo_pack": False, "has_endcard": False,
            }, "parse_source": "fake",
        }

    @staticmethod
    def _material(payload: dict[str, Any]) -> dict[str, Any]:
        count = min(payload.get("max_highlights", 5), 2)
        return {"highlights": [
            {"start": float(i * 3), "end": float(i * 3 + 2), "description": f"高光片段 {i + 1}",
             "tags": ["特写"], "recommended_for": ["Hook" if i == 0 else "展示卖点"]}
            for i in range(count)
        ]}

    @staticmethod
    def _script(payload: dict[str, Any]) -> dict[str, Any]:
        events = payload.get("video_analysis", {}).get("narrative_structure", {}).get("timeline_events", [])
        if not events:
            events = [{"start": 0, "end": 4, "event_name": "hook"}]
        shots = [
            {"index": i + 1, "block_index": i + 1, "start": float(e.get("start", i * 4)),
             "end": float(e.get("end", i * 4 + 4)), "narrative_stage": e.get("event_name", "product_detail"),
             "asset_source": "aigc_keyframe", "gap_codes": [], "fallback_applied": "aigc_keyframe",
             "requires_image_gen": True, "is_aigc_supplement": True,
             "stage_brief": e.get("description", "商品镜头")}
            for i, e in enumerate(events)
        ]
        blocks = [{"index": s["block_index"], "start": s["start"], "end": s["end"],
                   "render_mode": "I2V", "is_continuation": False, "shots": [s]} for s in shots]
        total = max(s["end"] for s in shots)
        manifest = {
            "schema_version": "script_manifest/v1", "version_type": payload.get("version_type"),
            "total_duration": total, "aspect_ratio": "9:16", "resolution": "1080x1920",
            "global_visual_anchor": payload.get("product", {}).get("visual_description", "商品视觉锚点"),
            "blocks": blocks, "cover_shot_index": 1, "gap_plan": payload.get("gap_plan", {"gaps": [], "resolutions": []}),
        }
        markdown = "\n".join(f"### 镜头 {s['index']}\n{s['stage_brief']}" for s in shots)
        return {"scriptMarkdown": markdown, "scriptManifest": manifest, "stitch_source": "fake"}

    @staticmethod
    def _polish(payload: dict[str, Any]) -> dict[str, Any]:
        product = dict(payload["product"])
        for field in payload["dirty_fields"]:
            value = product.get(field)
            if isinstance(value, str) and value.strip():
                product[field] = value.strip().rstrip("。") + "，表达更聚焦、具体且适合短视频呈现。"
        return {"product": product, "polish_source": "fake"}

    @staticmethod
    def _director(payload: dict[str, Any]) -> dict[str, Any]:
        message = payload["message"].strip()
        block = re.search(r"(?:重跑|重生成)\s*区块\s*(\d+)", message)
        if block:
            index = int(block.group(1))
            return {"reply": f"正在重跑区块 {index}…", "actions": [{"tool": "rerunBlock", "params": {"block_index": index}}]}
        mode = re.search(r"区块\s*(\d+).*?\b(r2v|i2v|t2v)\b", message, re.I)
        if mode:
            return {"reply": "已切换渲染模式。", "actions": [{"tool": "switchRenderModel", "params": {"block_index": int(mode.group(1)), "mode": mode.group(2).lower()}}]}
        return {"reply": f"AI 导演已收到：「{message}」。", "actions": []}
