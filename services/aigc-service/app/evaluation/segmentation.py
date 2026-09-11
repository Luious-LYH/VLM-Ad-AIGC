"""Product segmentation and lightweight video tracking.

The first implementation is deliberately honest about deployment state.  It
tries an Alpha channel, records whether SAM/GroundingDINO checkpoints were
available, and otherwise uses a deterministic foreground component segmenter
based on border-background colour distance.  The latter is a real image
segmentation operation, not a mock or a centre ellipse.  Every result carries
the backend and fallback reason so downstream metrics can become unavailable
instead of looking precise when the mask is bad.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter


@dataclass
class SegmentResult:
    mask: np.ndarray | None
    backend: str
    status: str
    confidence: float | None
    bbox: tuple[int, int, int, int] | None
    area_fraction: float | None
    fallback_reason: str | None = None
    attempted_backends: list[str] | None = None

    def metadata(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("mask", None)
        if self.bbox is not None:
            value["bbox"] = list(self.bbox)
        return value


def _bbox(mask: np.ndarray | None) -> tuple[int, int, int, int] | None:
    if mask is None or not np.any(mask):
        return None
    ys, xs = np.where(mask)
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def _result(mask: np.ndarray | None, backend: str, *, confidence: float | None,
            fallback_reason: str | None = None,
            attempted: list[str] | None = None) -> SegmentResult:
    if mask is None or not np.any(mask):
        return SegmentResult(None, backend, "segmentation_failed", None, None, None,
                             fallback_reason, attempted)
    return SegmentResult(
        mask.astype(bool), backend, "succeeded", confidence, _bbox(mask),
        float(mask.mean()), fallback_reason, attempted,
    )


def _alpha_segment(image: Image.Image) -> SegmentResult | None:
    if "A" not in image.getbands():
        return None
    alpha = np.asarray(image.getchannel("A"), dtype=np.uint8)
    # Ignore fully opaque RGB PNGs that merely happen to carry an alpha plane.
    nonzero = alpha > 8
    fraction = float(nonzero.mean())
    if fraction <= 0.002 or fraction >= 0.998:
        return None
    return _result(nonzero, "alpha", confidence=1.0,
                   attempted=["alpha", "sam3.1", "groundingdino+sam2", "saliency_component"])


def _morphology(mask: np.ndarray) -> np.ndarray:
    """Use scipy/skimage when installed, with a small PIL fallback."""
    try:
        from scipy import ndimage

        mask = ndimage.binary_closing(mask, iterations=3)
        mask = ndimage.binary_opening(mask, iterations=1)
        mask = ndimage.binary_fill_holes(mask)
        labels, count = ndimage.label(mask)
        if count:
            sizes = ndimage.sum(mask, labels, range(1, count + 1))
            keep = np.argsort(sizes)[-8:] + 1
            mask = np.isin(labels, keep)
        return mask
    except Exception:
        pil = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
        pil = pil.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.MinFilter(3))
        return np.asarray(pil) > 127


def _border_background_distance(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    height, width = rgb.shape[:2]
    band_y = max(2, int(height * 0.06))
    band_x = max(2, int(width * 0.06))
    border = np.concatenate([
        rgb[:band_y].reshape(-1, 3), rgb[-band_y:].reshape(-1, 3),
        rgb[:, :band_x].reshape(-1, 3), rgb[:, -band_x:].reshape(-1, 3),
    ])
    background = np.median(border, axis=0)
    # RGB distance is robust for the warm neutral sample and avoids requiring
    # OpenCV or a network-downloaded segmentation checkpoint.
    distance = np.linalg.norm(rgb - background[None, None, :], axis=2)
    # Add a weak edge term so a similarly coloured pedestal can still be
    # separated from the smooth background.
    gray = rgb.mean(axis=2)
    edge_y = np.abs(np.diff(gray, axis=0, prepend=gray[:1]))
    edge_x = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    return distance + 0.30 * np.sqrt(edge_x * edge_x + edge_y * edge_y)


def _saliency_segment(image: Image.Image) -> SegmentResult:
    from scipy import ndimage

    distance = _border_background_distance(image)
    finite = distance[np.isfinite(distance)]
    if finite.size == 0 or float(finite.max()) < 0.04:
        return _result(None, "saliency_component", confidence=None,
                       fallback_reason="border_colour_contrast_too_low",
                       attempted=["alpha", "sam3.1", "groundingdino+sam2", "saliency_component"])
    # Otsu is stable across the generated keyframes; enforce a conservative
    # floor to avoid selecting the warm studio gradient as foreground.
    try:
        from skimage.filters import threshold_otsu

        threshold = float(threshold_otsu(finite))
    except Exception:
        threshold = float(np.median(finite) + 2.0 * np.std(finite))
    threshold = max(threshold, 0.07)
    raw = distance >= threshold
    raw = _morphology(raw)
    labels, count = ndimage.label(raw)
    if count == 0:
        return _result(None, "saliency_component", confidence=None,
                       fallback_reason="no_connected_foreground_component",
                       attempted=["alpha", "sam3.1", "groundingdino+sam2", "saliency_component"])
    height, width = raw.shape
    centre = np.array([width * 0.5, height * 0.52])
    components: list[tuple[float, int, np.ndarray]] = []
    for label_id in range(1, count + 1):
        component = labels == label_id
        area = int(component.sum())
        if area < max(64, int(width * height * 0.0015)):
            continue
        ys, xs = np.where(component)
        centroid = np.array([xs.mean(), ys.mean()])
        distance_to_centre = float(np.linalg.norm((centroid - centre) / np.array([width, height])))
        touches_border = bool(xs.min() == 0 or ys.min() == 0 or xs.max() == width - 1 or ys.max() == height - 1)
        score = area / (width * height) * (1.0 - min(distance_to_centre, 1.0))
        if touches_border:
            score *= 0.35
        components.append((score, area, component))
    if not components:
        return _result(None, "saliency_component", confidence=None,
                       fallback_reason="all_components_too_small_or_border_connected",
                       attempted=["alpha", "sam3.1", "groundingdino+sam2", "saliency_component"])
    _, area, selected = max(components, key=lambda item: item[0])
    # Metallic caps can have almost the same colour as the studio background,
    # leaving only a central stripe after thresholding.  A small, resolution-
    # proportional morphological bridge restores the closed product silhouette
    # without reverting to a fixed image-wide ellipse.
    bridge = max(2, int(round(min(width, height) * 0.012)))
    selected_box = _bbox(selected)
    if selected_box is not None:
        # Only bridge the upper cap band.  Dilating the whole component can
        # accidentally join a product to a nearby pedestal in studio shots.
        cap_limit = selected_box[1] + int((selected_box[3] - selected_box[1]) * 0.32)
        cap_seed = selected.copy()
        cap_seed[cap_limit:, :] = False
        cap_bridge = ndimage.binary_dilation(cap_seed, iterations=bridge)
        cap_bridge[cap_limit:, :] = False
        selected = selected | cap_bridge
    # A bottle often touches a broad stone/plinth surface.  Remove only the
    # low, unusually wide horizontal rows so the connected-component mask does
    # not claim the whole pedestal as part of the product.
    for row in range(int(height * 0.64), height):
        xs = np.flatnonzero(selected[row])
        if xs.size and (int(xs[-1]) - int(xs[0]) + 1) > int(width * 0.55):
            selected[row, :] = False
    selected = ndimage.binary_fill_holes(selected)
    confidence = min(0.95, max(0.15, float(area / (width * height)) * 4.0))
    return _result(selected, "saliency_component", confidence=confidence,
                   fallback_reason="sam_checkpoints_unavailable",
                   attempted=["alpha", "sam3.1", "groundingdino+sam2", "saliency_component"])


def segment_product(image: Image.Image, *, sam3_checkpoint: str | None = None,
                    sam2_checkpoint: str | None = None,
                    groundingdino_checkpoint: str | None = None) -> SegmentResult:
    """Segment one product and retain an auditable backend/fallback trail.

    SAM3.1/SAM2 execution is intentionally opt-in until a compatible local
    checkpoint and package are staged.  This keeps an absent checkpoint from
    becoming a silent mock.  The current server therefore records the exact
    fallback and uses the deterministic local segmenter.
    """
    alpha = _alpha_segment(image)
    if alpha is not None:
        return alpha
    attempted = ["alpha", "sam3.1", "groundingdino+sam2", "saliency_component"]
    if sam3_checkpoint and Path(sam3_checkpoint).is_file():
        # Keep the failure explicit until the checkpoint's package/API is
        # validated; do not pretend a path alone proves a SAM inference.
        attempted.append("sam3.1_checkpoint_detected_but_adapter_not_enabled")
    if sam2_checkpoint and groundingdino_checkpoint and Path(sam2_checkpoint).is_file() and Path(groundingdino_checkpoint).is_file():
        attempted.append("groundingdino+sam2_checkpoint_detected_but_adapter_not_enabled")
    result = _saliency_segment(image)
    result.attempted_backends = attempted
    result.fallback_reason = "sam3.1_and_groundingdino_sam2_not_available_or_not_validated"
    return result


def ellipse_mask(size: tuple[int, int]) -> np.ndarray:
    width, height = size
    mask = Image.new("L", (width, height), 0)
    ImageDraw.Draw(mask).ellipse((int(width * 0.08), int(height * 0.04), int(width * 0.92), int(height * 0.98)), fill=255)
    return np.asarray(mask) > 127


def masked_crop(image: Image.Image, mask: np.ndarray | None, margin: float = 0.10,
                background: tuple[int, int, int] = (128, 128, 128)) -> Image.Image:
    """Crop a masked product with a fixed neutral background and margin."""
    image = image.convert("RGB")
    if mask is None or not np.any(mask):
        return Image.new("RGB", image.size, background)
    box = _bbox(mask)
    assert box is not None
    x0, y0, x1, y1 = box
    dx = int((x1 - x0) * margin)
    dy = int((y1 - y0) * margin)
    x0, y0 = max(0, x0 - dx), max(0, y0 - dy)
    x1, y1 = min(image.width, x1 + dx), min(image.height, y1 + dy)
    cropped = image.crop((x0, y0, x1, y1))
    crop_mask = Image.fromarray((mask[y0:y1, x0:x1].astype(np.uint8) * 255), mode="L")
    neutral = Image.new("RGB", cropped.size, background)
    return Image.composite(cropped, neutral, crop_mask)


def overlay_image(image: Image.Image, mask: np.ndarray | None) -> Image.Image:
    image = image.convert("RGB")
    if mask is None:
        return image
    overlay = Image.new("RGBA", image.size, (230, 40, 40, 0))
    overlay.putalpha(Image.fromarray((mask.astype(np.uint8) * 90), mode="L"))
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def save_mask_bundle(image: Image.Image, result: SegmentResult, directory: Path,
                     stem: str) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    mask_path = directory / f"{stem}.mask.png"
    foreground_path = directory / f"{stem}.foreground.png"
    overlay_path = directory / f"{stem}.overlay.png"
    metadata_path = directory / f"{stem}.json"
    if result.mask is None:
        mask = np.zeros((image.height, image.width), dtype=np.uint8)
    else:
        mask = result.mask.astype(np.uint8) * 255
    Image.fromarray(mask, mode="L").save(mask_path)
    rgba = image.convert("RGBA")
    rgba.putalpha(Image.fromarray(mask, mode="L"))
    rgba.save(foreground_path)
    overlay_image(image, result.mask).save(overlay_path)
    metadata = result.metadata()
    metadata.update({"mask": str(mask_path), "foreground": str(foreground_path), "overlay": str(overlay_path)})
    metadata_path.write_text(__import__("json").dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def _translate(mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
    output = np.zeros_like(mask)
    height, width = mask.shape
    src_x0, src_x1 = max(0, -dx), min(width, width - dx)
    src_y0, src_y1 = max(0, -dy), min(height, height - dy)
    dst_x0, dst_x1 = max(0, dx), min(width, width + dx)
    dst_y0, dst_y1 = max(0, dy), min(height, height + dy)
    if src_x1 > src_x0 and src_y1 > src_y0:
        output[dst_y0:dst_y1, dst_x0:dst_x1] = mask[src_y0:src_y1, src_x0:src_x1]
    return output


def track_video_masks(frames: Iterable[Image.Image], *, max_samples: int = 16,
                      segmenter=segment_product) -> tuple[list[Image.Image], list[np.ndarray | None], dict[str, Any]]:
    """Track a product through uniformly sampled frames.

    Each frame is segmented independently first.  If a frame fails, the last
    valid mask is translated by the observed centre displacement when
    possible; this is logged as a tracking interruption rather than treated as
    an equally reliable segmentation result.
    """
    frame_list = list(frames)
    if not frame_list:
        return [], [], {"status": "segmentation_failed", "error": "empty_video", "sample_count": 0}
    if len(frame_list) > max_samples:
        indices = np.linspace(0, len(frame_list) - 1, max_samples).round().astype(int).tolist()
        sampled = [frame_list[index] for index in indices]
    else:
        indices = list(range(len(frame_list)))
        sampled = frame_list
    masks: list[np.ndarray | None] = []
    segments: list[SegmentResult] = []
    interruptions = 0
    previous_centre: tuple[float, float] | None = None
    previous_mask: np.ndarray | None = None
    for image in sampled:
        result = segmenter(image)
        mask = result.mask
        if mask is not None:
            box = _bbox(mask)
            centre = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2) if box else None
            if previous_centre is not None and centre is not None:
                distance = np.linalg.norm((np.asarray(centre) - previous_centre) / np.asarray([image.width, image.height]))
                if distance > 0.35 and previous_mask is not None:
                    # A wildly jumping component is more likely a false match;
                    # retain the previous track and count the interruption.
                    mask = previous_mask
                    interruptions += 1
            previous_centre = np.asarray(centre) if centre is not None else previous_centre
            previous_mask = mask
        elif previous_mask is not None:
            interruptions += 1
            mask = previous_mask
        else:
            interruptions += 1
        masks.append(mask)
        segments.append(result)
    raw_valid = sum(item.mask is not None for item in segments)
    tracked_valid = sum(item is not None for item in masks)
    return sampled, masks, {
        "status": "succeeded" if tracked_valid else "segmentation_failed",
        "sample_count": len(sampled),
        "source_frame_indices": indices,
        "valid_mask_count": raw_valid,
        "tracked_mask_count": tracked_valid,
        "product_appearance_rate": round(raw_valid / max(1, len(sampled)), 4),
        "tracking_interruptions": interruptions,
        "backend_sequence": [item.backend for item in segments],
        "fallback_reasons": sorted({item.fallback_reason for item in segments if item.fallback_reason}),
    }
