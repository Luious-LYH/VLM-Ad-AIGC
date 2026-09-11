from pathlib import Path

import pytest

from app.config import Settings
from app.security import UnsafeAssetUri, validate_asset_uri


def config(tmp_path: Path) -> Settings:
    return Settings(
        asset_roots=(tmp_path.resolve(),),
        url_allowlist=("cdn.example.com", "*.assets.example.com"),
        allow_data_urls=True,
        max_data_url_bytes=32,
    )


def test_url_allowlist_and_credentials(tmp_path):
    cfg = config(tmp_path)
    assert validate_asset_uri("https://cdn.example.com/a.png", cfg).startswith("https://")
    assert validate_asset_uri("https://x.assets.example.com/a.png", cfg).startswith("https://")
    with pytest.raises(UnsafeAssetUri):
        validate_asset_uri("https://evil.example/a.png", cfg)
    with pytest.raises(UnsafeAssetUri):
        validate_asset_uri("https://cdn.example.com.evil.example/a.png", cfg)
    with pytest.raises(UnsafeAssetUri):
        validate_asset_uri("https://user:pass@cdn.example.com/a.png", cfg)


def test_path_traversal_and_encoding(tmp_path):
    cfg = config(tmp_path)
    safe = tmp_path / "nested" / "asset.png"
    assert validate_asset_uri(str(safe), cfg) == str(safe.resolve(strict=False))
    with pytest.raises(UnsafeAssetUri):
        validate_asset_uri(str(tmp_path / ".." / "secret.txt"), cfg)
    with pytest.raises(UnsafeAssetUri):
        validate_asset_uri(str(tmp_path) + "/%2e%2e/secret.txt", cfg)
    with pytest.raises(UnsafeAssetUri):
        validate_asset_uri("../asset.png", cfg)


def test_data_url_is_bounded_and_typed(tmp_path):
    cfg = config(tmp_path)
    assert validate_asset_uri("data:image/png;base64,aGVsbG8=", cfg).startswith("data:image/png")
    with pytest.raises(UnsafeAssetUri):
        validate_asset_uri("data:text/html;base64,aGVsbG8=", cfg)
    with pytest.raises(UnsafeAssetUri):
        validate_asset_uri("data:image/png;base64," + "YQ==" * 30, cfg)


def test_http_rejection_happens_at_endpoint(client):
    response = client.post("/v1/reference/analyze", json={"video_url": "https://evil.example/video.mp4"})
    assert response.status_code == 422

