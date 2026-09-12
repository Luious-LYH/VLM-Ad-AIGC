# Project layout

`VLM-Ad-AIGC` is the standalone model-and-evaluation project extracted from the
original MetaCut UI repository. The folders are separated by responsibility so
the full path from an input product image to an evaluated video is easy to
reproduce.

```text
configs/
  v0.1/                  versioned model registry and evaluation rules
  *comparison.json       reproducible 2x2x2 / Phase 2.5 matrices
scripts/
  run-phase2-*.py        VLM, keyframe and video generation workers
  run-v01-*.py           end-to-end orchestration and evaluation
  run-phase3-*.py        identity-aware ablation runner
  report/evaluate/*.py   metrics, reports, smoke tests and transcode helpers
services/aigc-service/
  app/                   FastAPI schemas, adapters, local backends and metrics
  tests/                 API, schema, segmentation and fallback tests
samples/sample-02/
  input.png              curated product input
  keyframes/             one keyframe per image-generation backend
  videos/                four unique image-model x video-model MP4s
  records/               generation provenance sidecars
results/                 local-only reports and Phase 3 artifacts (ignored)
runs/                    server/local run manifests, masks, evidence and logs (ignored)
third_party/             local runtime sources such as LTX-Video (ignored)
```

Model weights, virtual environments, caches and generated run data are never
committed. On the authorized server they live under the shared model root
`/public/lyh/projects/CV_projects/models` and the project runtime/output roots
under `/public/lyh/projects/CV_projects/VLM-Ad-AIGC`.

The four curated MP4s are intentionally stored once. VLM choices affect the
structured understanding/evaluation branch, not the diffusion request, so the
same MP4 is referenced by both VLM rows in the comparison table.
