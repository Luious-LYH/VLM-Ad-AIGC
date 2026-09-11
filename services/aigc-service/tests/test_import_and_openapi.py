import json
import subprocess
import sys


def test_import_does_not_load_heavy_ml_frameworks():
    code = "import sys; import app.main; print(json.dumps(sorted(set(sys.modules) & {'torch','transformers','diffusers'})))"
    result = subprocess.run(
        [sys.executable, "-c", "import json; " + code], capture_output=True, text=True, check=True
    )
    assert json.loads(result.stdout.strip()) == []


def test_openapi_contains_all_contract_routes(client):
    spec = client.get("/openapi.json").json()
    expected = {
        "/healthz", "/v1/capabilities", "/v1/reference/analyze", "/v1/product/parse",
        "/v1/material/analyze", "/v1/script/stitch", "/v1/product/polish",
        "/v1/director/intent", "/v1/image/generations", "/v1/video/generations",
        "/v1/jobs/{job_id}",
    }
    assert expected <= set(spec["paths"])

