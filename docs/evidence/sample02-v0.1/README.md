# sample-02 v0.1 evidence

These small JSON files are a checked-in index of the real server run used in
the README tables. The full masks, overlays, evidence frames, logs and video
sidecars remain on the authorized GPU server because they are generated run
data and are intentionally excluded from Git.

- `manifest.json`: input SHA256, prompt hash, model registry, hardware snapshot,
  segmentation policy and the four evaluated video records.
- `metrics-summary.json`: per-video masked-DINO, CLIP, temporal, storyboard,
  segmentation and engineering metrics.
- `qwen-product.json` / `internvl-product.json`: the two real VLM product
  understanding outputs and their schema/attribute/OCR metrics.
- `phase3-ablation.json` / `phase3-ablation.md`: offline mask, sampling and
  Storyboard ablation computed from the same four videos.
- `pipeline-wrapper-live-manifest.json` / `pipeline-wrapper-live-metrics.json`:
  the independent one-command `--generate` smoke run. It generated a fresh
  FLUX.2-klein keyframe and Wan2.2 video, then completed mask-aware evaluation
  in the same CLI invocation.

The product attributes used for comparison are manually reviewed labels for
this single demonstration image, not a public dataset annotation. DINO, CLIP,
temporal and storyboard values are reference-preservation or visual-rule
proxies; they do not establish general model superiority or human aesthetic
quality. The server path recorded in `manifest.json` is:
`/public/lyh/projects/CV_projects/VLM-Ad-AIGC/runs/v0.1/sample02-fresh-provenance-v3/`.
