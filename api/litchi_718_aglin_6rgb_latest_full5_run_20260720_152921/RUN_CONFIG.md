# 718-aglin Stage 1-4 annotation run

- Run ID: `litchi_718_aglin_6rgb_latest_full5_run_20260720_152921`
- Inputs: 6 complete RGB videos, 1920x1080, 30 fps, 14,498 original frames
- Source policy: source videos are not modified; unique input names are derived from their recording directories
- Code: `api_seed_openai_easy_no_short_step`, latest local/remote hash-matched copy
- Model: `doubao-seed-2-1-pro-260628`
- Ark video preprocessing: 5.0 fps
- Dense frame-index timeline: 5.0 fps
- Frame convention: original-video frames are 0-based; end-frame fields are explicitly inclusive/exclusive
- Stages 1-4: completed locally with 2-way video parallelism and resumable retries
- Stage 5: disabled by user request; no Stage 5 output is retained
- Stage 5 hands: WiLoR on CPU, primary-center-operator filtering
- Stage 5 grounding/tracking: SAM3 on CUDA, conditional Ark hand/object grounding
- Stage 5 acceptance: `quality_gate.status == passed` and `eligible_for_automatic_use == true`
- Stage 5 schema: `stage5-bbox-v9-bounded-visibility-evidence`
- Source manifest: `INPUT_MANIFEST.tsv`

The API key is not copied into this run directory or command line. The pipeline's
validated default key is passed internally to Stage 5 and is intentionally omitted
from logs and manifests.

## Stage 5 pause

On 2026-07-20 the requested scope changed to Stage 1-4 only. The automatic Stage 5
handoff had started but had completed zero videos: there were no
`stage5_summary.json` files. The Stage 5 process group was stopped, its two partial
output directories were removed locally and remotely, and `.stage5_disabled` was
installed as a hard launch guard. The six audited Stage 1-4 outputs are authoritative.
