from pathlib import Path

from app.adapters.registry import get_model, load_registry
from app.evaluation.segmentation import segment_product
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_v01_registry_contains_all_runtime_roles():
    registry = load_registry(PROJECT_ROOT / "configs/v0.1/model-registry.json")
    assert registry["schema_version"] == "model_registry/v1"
    assert get_model(registry, "vlm", "qwen3_vl")["path_env"] == "AIGC_VLM_PATH"
    assert get_model(registry, "image", "flux2_klein")["model_id"]
    assert get_model(registry, "video", "wan22_ti2v_5b")["capabilities"]
    assert get_model(registry, "segmentation", "saliency_component")["status"] == "validated_on_server"


def test_segmentation_metadata_is_versioned():
    result = segment_product(Image.new("RGB", (16, 16), (220, 220, 220)))
    assert result.metadata()["schema_version"] == "segmentation/v1"
