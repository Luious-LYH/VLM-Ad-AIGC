# Experiment Audit Report

**Date**: 2026-09-12  
**Project**: VLM-Ad-AIGC v0.1  
**Auditor**: independent fallback reviewer (`gpt-5.6-terra`, high reasoning)

## Overall verdict: WARN (initial review FAIL remediated for shipped claims)

The preferred GPT-5.4 reviewer endpoint was unavailable (HTTP 503 account-pool
exhaustion). A separate model-family reviewer completed a read-only audit. The
initial FAIL concerned stale server-only references and unlabeled proxy scope;
the repository now ships an evidence index, labels manual labels/proxy metrics
explicitly, and no longer treats Hunyuan as a v0.1 result. Full generated runs
remain on the authorized server and are intentionally not committed.

## Checks

### A. Ground-truth provenance: WARN

The product attributes in `configs/phase2-comparison.json` and the sample
records are manually reviewed labels for one demonstration image, not a
dataset annotation. DINO compares the input/reference product to generated
regions. These are now explicitly called manual labels and reference-preserving
proxies in `README.md` and `docs/evidence/sample02-v0.1/README.md`; no candidate
video is used as hidden ground truth.

### B. Score normalization: PASS

DINO/CLIP vectors are L2-normalized before cosine similarity. Temporal and
visual scores use fixed thresholds and frame-count aggregates, not a
model-dependent maximum or minimum. The metrics are bounded heuristics and are
not presented as calibrated quality scores.

### C. Result-file existence and claim matching: WARN

The shipped v0.1 table is backed by the checked-in evidence index:
`docs/evidence/sample02-v0.1/manifest.json`, `metrics-summary.json`,
`qwen-product.json`, and `internvl-product.json`. The four DINO/stability/
Storyboard values and the two VLM summaries can be checked there. Historical
multi-sample Phase-2 reports under ignored `results/` are not used as the
README's single-sample claim. Full masks, overlays and logs are server-only and
their exact path is recorded in the evidence README.

### D. Dead code: WARN

Core DINO, CLIP, segmentation, temporal and event-level Storyboard functions
are called by `scripts/run-v01-evaluation.py` and serialized into metrics. The
legacy `save_evidence` helper is redundant with the inline evidence writer and
can be removed in a later cleanup; it does not affect reported values.

### E. Scope: WARN

The demonstrated v0.1 run covers one product, one seed, two VLMs and four
unique image/video combinations. VLM rows intentionally reuse the same MP4
because VLM does not alter the diffusion request. The README does not claim
statistical significance, broad robustness or general model superiority.

### F. Evaluation type: WARN

| Evaluation | Type | Qualification |
|---|---|---|
| VLM attributes/OCR | `synthetic_proxy` with manual labels | useful for this demo image only |
| DINO product consistency | `self_supervised_proxy` | reference-preservation, not visual quality |
| CLIP text match | `synthetic_proxy` | fixed authored caption |
| Temporal diagnostics | `self_supervised_proxy` | frame-difference/region rules |
| Storyboard rules | `synthetic_proxy` | authored event windows plus visual rules |
| Human review | `human_eval` | required for final visual acceptance; not replaced by metrics |

## Claim impact

- Four unique model-combination videos and their server-side evaluation: **supported**.
- VLM structured understanding and OCR/schema smoke tests: **supported for the sample**.
- DINO/CLIP/temporal/Storyboard values: **supported with proxy qualifier**.
- General superiority, cross-dataset robustness, or statistical significance: **unsupported**.
- HunyuanVideo as a v0.1 result: **not claimed**; its noisy output is documented as excluded.
