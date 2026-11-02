# Stage 5 SAM3 backend

Stage 5 now defaults to `--bbox-backend sam3`.

## Data flow

1. Decode every original video frame without temporal subsampling.
2. Run the local WiLoR/Ultralytics detector for all anatomical `left_hand` and
   `right_hand` candidates, then retain only one temporally consistent primary pair
   whose sustained manipulation centroid is closest to the image center.
3. Run SAM3 once for the `hand` concept and once for each manipulated-object label from
   Stage 3 `hand_object_relations`.
4. Build one persistent primary-instance chain for each anatomical hand using
   sequence-wide WiLoR evidence, one-to-one anchor assignment, and strict temporal
   continuity. Only one chain instance may be active per hand in a frame, while a
   verified non-overlapping fragment may continue the same hand after occlusion.
5. Select the correct SAM3 object instance using active step intervals and hand-object
   proximity/contact evidence.
6. If an expected glove-covered hand is missing/weak, call Ark image grounding on a
   small number of target frames with previous/next identity-context images, then
   track accepted API anchors only inside the weak interval. A structured
   `not_visible` response with the explicit reason `occluded`, `out_of_frame`, or
   `absent` is preserved as bounded negative visibility evidence rather than being
   misreported as a failed API call. One API target can confirm at most 0.5 seconds;
   unsupported portions of a long gap remain unexplained and force review.
7. If a manipulated-object track is unresolved, sparse, or low-confidence, call Ark
   image grounding for a corrective anchor, reject anchors inconsistent with the
   primary hands/SAM3, and fill weak active intervals with interval-scoped optical flow.
8. Write one record for every original 0-based frame with pixel `xyxy`, normalized
   `[0,999]` `xyxy`, timestamp, visibility, confidence, source, and active step IDs.

## Primary-operator scope

All five stages annotate only the primary center operator. Stage 1-4 prompts ignore
actions, objects, transitions, and boundaries produced solely by secondary edge or
background hands. Stage 5 keeps every WiLoR candidate until pair selection, scores
pair-center proximity plus temporal continuity, and discards secondary candidates.
Once a primary identity is established, a hard short-horizon motion gate rejects
impossible hand switches. SAM3 starts with one-to-one anchor assignment. A
non-overlapping fragment may join the same hand chain only after strict short-gap
continuity validation; simultaneous instances cannot compete for one hand, and an
unselected edge instance cannot replace the active locked fragment. Segment-boundary
continuity carries identity across SAM3 chunks. Ark hand anchors use
previous/target/next images and must
also pass local center/temporal validation. Ark object grounding is restricted to
objects manipulated by the selected pair. Selection, assignment, rejection, and
identity-pruning evidence is stored in `qa_summary.json`.

If left/right boxes collapse onto the same physical box, Stage 5 resolves the duplicate
using the current step relation first and source/confidence evidence second. It never
emits two near-identical hand boxes merely to satisfy both labels.

SAM3 runs in its dedicated Conda environment through `stage5_sam3_backend.py`; the
pipeline process remains isolated in `.venv_bbox_tracking`. A CPU WiLoR run cannot
leak its `CUDA_VISIBLE_DEVICES=-1` setting into the SAM3 subprocess.

## Prepare the server

```bash
cd /home/descfly/luzheng/ego_data_visualization/api/api_seed_openai_easy_no_short_step/luzheng1/Qwen-PC/ECCV/four_stage_100frames_refine_v2
./setup_stage5_bbox_env.sh
```

The setup script reuses the validated checkpoint instead of copying 3.45 GB:

```text
/home/descfly/luzheng/ego_data_visualization/api/models/sam3/sam3.pt
  -> /home/descfly/caoheng/sam3/ckpt/sam3.pt
```

It writes `api/models/sam3/MODEL_MANIFEST.json` with the resolved path, byte size,
SHA-256, BPE vocabulary, WiLoR hand detector, SAM3 repository commit, and Python
interpreter. The API-managed paths are:

```text
api/models/sam3/sam3.pt
api/models/sam3/bpe_simple_vocab_16e6.txt.gz
api/models/wilor/detector.pt
```

## Run

Include Stage 5 in the normal pipeline stage list. The SAM3 paths are discovered by
default, so no extra model arguments are required after setup.

On the server, run the main pipeline with the validated environment rather than the
bare system `python3` (which may not contain the OpenAI SDK):

```bash
/home/descfly/luzheng/ego_data_visualization/api/.venv_bbox_tracking/bin/python pipeline.py ...
```

Useful explicit arguments:

```text
--bbox-backend sam3
--sam3-threshold 0.5
--sam3-chunk-size 500
--sam3-max-instances 8
--sam3-hand-fusion
--api-hand-assist
--api-hand-max-calls-per-step 3
--api-hand-max-total-calls 24
--api-object-min-track-coverage 0.60
--api-object-min-mean-confidence 0.55
--api-grounding-thinking enabled
--api-grounding-reasoning-effort medium
```

Ark assistance is conditional: good SAM3/WiLoR tracks make zero Stage 5 API calls.
The API is used only when expected hands or manipulated objects fail the configured
quality gates. Disable either assistance path explicitly with:

```text
--no-api-hand-assist
--no-api-grounding
```

Stage 5 Ark grounding defaults to enabled thinking with medium reasoning effort. The
request sends `thinking.type=enabled` and `reasoning_effort=medium` through the OpenAI
SDK-compatible Chat Completions API.

Grounding DINO plus SAM2 remains a separate explicit compatibility path:

```text
--sam3-legacy-fallback
```

Without that flag, unresolved objects first receive direct Ark assistance, then Stage 5
fails if neither SAM3 nor Ark can establish a reliable object anchor unless
`--allow-unresolved-objects` is explicitly supplied.

## Outputs

The normal Stage 5 files remain compatible:

- `stage5/frame_bboxes.jsonl`
- `stage5/tracks.json`
- `stage5/qa_summary.json`
- `stage5/stage5_summary.json`
- `stage5/bbox_overlay.mp4`

`qa_summary.json` includes per-track active-interval bbox coverage, longest missing
run, direct-observation coverage, confidence, center jump, zero-IoU jumps, area-ratio
changes, SAM3 instance switches, optical-flow details, API rejection reasons, and a
quality gate. Only `passed` outputs are marked `eligible_for_automatic_use`; warning
or failed outputs require review. Tracked/interpolated boxes are explicitly marked as
estimated rather than observed.

Only a `passed` result with `eligible_for_automatic_use=true` can be resumed. Warning
or failed outputs are always regenerated instead of being accepted merely because all
expected files exist.

Confirmed absent/out-of-frame/occluded windows retain `bbox=null`, their visibility
reason, and source `ark_hand_visibility_confirmation`. They remain visible in coverage
metrics but are excluded from `longest_unexplained_missing_frames`. An API response
with reason `unknown` is rejected as insufficient evidence and cannot suppress QA.

SAM3 also adds reproducibility artifacts:

- `stage5/sam3_requests.json`
- `stage5/sam3_candidates.json`

The schema version is `stage5-bbox-v9-bounded-visibility-evidence`, preventing older
outputs without bounded negative-visibility evidence and strict identity QA from being
resumed.

Run the dependency-light regression suite with:

```bash
python3 -m unittest discover -s tests -v
```
