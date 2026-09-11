ASSET = "https://assets.example.test/product.mp4"
IMAGE = "https://assets.example.test/product.png"


def test_health_and_capabilities(client):
    assert client.get("/healthz").json()["status"] == "ok"
    capabilities = client.get("/v1/capabilities").json()
    assert capabilities["backend"] == "fake"
    assert capabilities["models"]["vlm"]["loaded"] is False


def test_all_sync_endpoints(client):
    reference = client.post("/v1/reference/analyze", json={"video_url": ASSET})
    assert reference.status_code == 200
    analysis = reference.json()["result"]
    assert analysis["schema_version"] == "video_analysis/v1"

    product = client.post("/v1/product/parse", json={
        "product_description": "星云保温杯 高颜值随行杯", "product_image_urls": [IMAGE]
    })
    assert product.status_code == 200
    parsed = product.json()["result"]
    assert parsed["schema_version"] == "product/v1"

    material = client.post("/v1/material/analyze", json={"video_url": ASSET})
    assert material.status_code == 200
    assert material.json()["result"]["highlights"]

    stitch = client.post("/v1/script/stitch", json={
        "video_analysis": analysis, "product": parsed,
    })
    assert stitch.status_code == 200
    assert stitch.json()["result"]["scriptManifest"]["schema_version"] == "script_manifest/v1"

    polish = client.post("/v1/product/polish", json={
        "product": parsed, "dirty_fields": ["visual_description"]
    })
    assert polish.status_code == 200
    assert polish.json()["result"]["polish_source"] == "fake"

    director = client.post("/v1/director/intent", json={"message": "重跑区块 2"})
    assert director.status_code == 200
    assert director.json()["result"]["actions"][0]["tool"] == "rerunBlock"

    for response in (reference, product, material, stitch, polish, director):
        run = response.json()["run"]
        assert run["model_id"]
        assert run["model_revision"]
        assert run["run_id"].startswith("run_")


def test_sync_validation(client):
    assert client.post("/v1/product/parse", json={}).status_code == 422
    assert client.post("/v1/product/polish", json={
        "product": {}, "dirty_fields": ["admin"]
    }).status_code == 422

