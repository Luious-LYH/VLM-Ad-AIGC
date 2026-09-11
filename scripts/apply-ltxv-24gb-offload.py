"""Apply the small 24 GB GPU compatibility overlay to a staged LTX-Video checkout.

The vendor checkout is intentionally ignored by Git.  Keeping this idempotent
overlay in the project makes the server deployment reproducible without
committing the upstream model/runtime sources.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument(
    "--root",
    type=Path,
    default=Path("third_party/LTX-Video"),
    help="staged LTX-Video checkout",
)
args = parser.parse_args()

pipeline = args.root / "ltx_video/pipelines/pipeline_ltx_video.py"
vae_encode = args.root / "ltx_video/models/autoencoders/vae_encode.py"
if not pipeline.is_file() or not vae_encode.is_file():
    raise SystemExit(f"LTX-Video checkout not found below {args.root}")


def replace_once(path: Path, old: str, new: str) -> bool:
    text = path.read_text(encoding="utf-8")
    if new in text:
        return False
    if text.count(old) != 1:
        raise SystemExit(f"expected one patch anchor in {path}: {old!r}")
    path.write_text(text.replace(old, new), encoding="utf-8")
    return True


changed = 0
changed += replace_once(
    pipeline,
    "import torch\n",
    "import torch\nfrom accelerate.hooks import remove_hook_from_module\n",
)
changed += replace_once(
    pipeline,
    ").to(dtype=init_latents.dtype)\n",
    ").to(dtype=init_latents.dtype, device=init_latents.device)\n",
)
pipeline_text = pipeline.read_text(encoding="utf-8")
if "remove_hook_from_module(self.vae, recurse=True)" not in pipeline_text:
    updated, count = re.subn(
        r"            if offload_to_cpu and str\(latents\.device\)\.startswith\(\"cuda\"\):\n                self\.vae\.to\(latents\.device\)\n",
        "            if str(latents.device).startswith(\"cuda\"):\n                # The custom causal VAE is not fully covered by Diffusers hooks.\n                remove_hook_from_module(self.vae, recurse=True)\n                self.vae.to(latents.device)\n",
        pipeline_text,
    )
    if count != 1:
        raise SystemExit("expected one VAE offload block in pipeline_ltx_video.py")
    pipeline.write_text(updated, encoding="utf-8")
    changed += 1
for old, new in (
    (
        "vae.mean_of_means.to(latents.dtype)",
        "vae.mean_of_means.to(device=latents.device, dtype=latents.dtype)",
    ),
    (
        "vae.std_of_means.to(latents.dtype)",
        "vae.std_of_means.to(device=latents.device, dtype=latents.dtype)",
    ),
):
    changed += replace_once(vae_encode, old, new)

print(f"LTX-Video 24 GB overlay: {'updated' if changed else 'already applied'} ({changed} edits)")
