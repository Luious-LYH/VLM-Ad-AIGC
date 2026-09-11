# VLM-Ad-AIGC Local AIGC Service

A lightweight FastAPI boundary between the Next.js workbench and local GPU
models. The dependency-light `fake` backend validates API integration, job
state, provenance, idempotency, and security without CUDA. `AIGC_BACKEND=local`
uses lazy imports and staged local weights for Qwen3-VL, SDXL + IP-Adapter,
FLUX.2-klein, LTXV, Wan2.2 and HunyuanVideo-1.5. An unavailable experimental backend fails its
own job with an actionable message; it never silently calls a paid API.
Model-to-model fallback is disabled by default; set `AIGC_ALLOW_EXPLICIT_FALLBACK=true`
only for a documented fallback run, whose response records the requested and
actual model IDs plus the fallback reason.

## Quick start

```bash
cd services/aigc-service
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
uvicorn app.main:app --host 127.0.0.1 --port 8100
pytest
```

On Windows PowerShell activate with `.venv\\Scripts\\Activate.ps1`. OpenAPI is
available at `http://127.0.0.1:8100/docs`.

## API surface

Synchronous, schema-validated routes:

- `POST /v1/reference/analyze`
- `POST /v1/product/parse`
- `POST /v1/material/analyze`
- `POST /v1/script/stitch`
- `POST /v1/product/polish`
- `POST /v1/director/intent`

Queued routes return HTTP 202 and a status URL:

- `POST /v1/image/generations` (compatibility endpoint)
- `POST /v1/keyframes` (`sdxl_ip_adapter` or `flux2_klein`)
- `POST /v1/video/generations` (compatibility endpoint)
- `POST /v1/video` (`ltxv_2b`, `wan22_ti2v_5b` or `hunyuanvideo_15_i2v`)
- `GET /v1/jobs/{id}`

Supply `Idempotency-Key` on generation requests. Repeating the same request
returns the existing job; reusing a key with a different body returns HTTP 409.
The fake backend supports the exact prompt token `__fake_fail__` to exercise a
failed job in development tests.

Every sync response and queued job contains model ID/revision, backend, run ID,
seed, latency, and determinism metadata. Fake artifacts are intentionally URLs
only; the backend does not write generated files.

## Asset security

Remote assets are rejected unless their exact host (or an explicit `*.domain`
pattern) appears in `AIGC_URL_ALLOWLIST`. URL credentials, unsupported schemes,
and fragments are rejected. Local paths must be absolute and resolve beneath an
`AIGC_ASSET_ROOTS` directory, including percent-decoded traversal checks. Bounded
image/video data URLs may be enabled for trusted internal calls.

Set `AIGC_API_TOKEN` when the service is reachable outside localhost. Clients
must then send `Authorization: Bearer <token>`. Do not expose this service or GPU
workers directly to the public internet.

## Local model environment

The base install stays CPU-only. Extras separate deployment roles so each GPU
worker installs only what it needs:

```bash
pip install -e '.[vlm]'
pip install -e '.[image]'
pip install -e '.[video]'
pip install -e '.[hunyuan]'
pip install -e '.[queue]'
```

The HunyuanVideo-1.5 worker requires Diffusers 0.40 because
`HunyuanVideo15ImageToVideoPipeline` is not present in older releases. The
production server exposes this isolated runtime through `AIGC_PYTHONPATH` so
the Qwen worker's dependency pins remain untouched.

Set `AIGC_BACKEND=local`, use this project's `public/uploads` directory as both
`AIGC_ASSET_ROOTS` and `AIGC_OUTPUT_ROOT`, then set the model path variables
shown in `.env.example`. The local adapter records model ID, revision,
seed, latency and peak VRAM for every image/video result.

For the initial 3×3090 deployment, run one worker process per selected model
backend and bind workers through `CUDA_VISIBLE_DEVICES` or `AIGC_DEVICE_*`.
Keep the FastAPI gateway on loopback. The in-memory queue is deliberately
single-process; replace `JobStore` with Redis/RQ only after the demo is stable.

HunyuanVideo-1.5 uses an isolated two-process candidate runner: one 480p
step-distilled candidate on each of `AIGC_HY15_GPUS` (default `1,2`), then a
lightweight temporal-stability ranker selects the returned MP4. This is the
intentional multi-GPU path for a single queued request; the older LTX/Wan
paths remain single-card fallbacks.
