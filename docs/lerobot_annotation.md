# LeRobot input and output: phase 1 implementation

The user-selected contract is LeRobot v2.1 in, LeRobot v2.1 out, with per-frame
subtask text. The industrial PC performs conversion and annotation locally.
The final dataset can be uploaded to separately configured data/training receivers.
No manual interaction is part of the run command. A failed quality check is an
automatic failure/retry outcome, not a human-approval step or a false success.

## Delivery contract

```
annotated_dataset/
  data/chunk-000/episode_000000.parquet
  videos/chunk-000/observation.images.cam{0,1,2}/episode_000000.mp4
  meta/info.json
  meta/tasks.jsonl
  meta/episodes.jsonl
  meta/episodes_stats.jsonl
```

Original episode count, lengths, numbering, global indices, timestamps, states,
actions, engagement flags, depth/audit columns, camera layout, Arrow schema and
video bytes are preserved. Only `task_index` Parquet VALUES are changed. The
task vocabulary, episode task lists, total_tasks and task_index statistics are
updated consistently. An input-specific scalar episode `task_index` is omitted
for a multi-task episode: standard v2.1 uses episode `tasks` and per-row task_index.
Identical format means loader/schema compatibility, not identical file bytes.
Parquet is reencoded; all non-task values and schema metadata are compared.

Original conversion/ingest verification documents are NOT copied into delivery:
their hashes describe the unannotated dataset and would become stale. Source
tasks, metadata and annotation provenance are retained separately in work/.
Rich object/bbox labels are not part of this phase's unchanged feature schema.
They need an optional future schema extension or work-side artifact interface.

## Commands

Run from the repository root, with Python 3.12, ffmpeg/ffprobe installed:

```bash
python -m pip install -r requirements_annotation.txt
python -m lerobot_annotation inspect --input /local/cleaned_lerobot \
  --report /local/annotation-work/preflight.json
```

This is a metadata/schema/artifact check, not full video decoding or full sensor
validation. Both `0_127_data_lerobot` (format reference) and `209_213_data_lerobot`
(current test material) have v2.1 layout. Do not write into their remote mount.

For actual annotation:

```bash
export ARK_API_KEY='set securely in the industrial-PC environment'
export MODEL_NAME='an account-authorized Ark video model'
python -m lerobot_annotation run \
  --input /local/cleaned_lerobot \
  --output /local/delivery/annotated_lerobot \
  --work /local/annotation-work/run-001 \
  --window-seconds 30 --video-fps 5 --attempts 2 --max-api-calls 300
```

Input, output and work must be separate disjoint directories. Output must be NEW.
The job reads cam0 (user-designated primary view), checks each decoded window,
calls the existing Ark client and automatically validates complete chronological
step coverage. Window-local seconds are converted to source-frame ranges.
Boundaries remain VLM estimates, not frame-accurate verified event labels.
cam1/cam2 bytes are preserved; multi-view inference is a later phase.

Idle/repositioning can be labeled. Unknown laterality is expressed in text.
Uncertain steps, gaps, out-of-range times or invalid JSON cause bounded automatic
retry, then a failed job. No partial dataset is published or uploaded.
Validated window results are cached by source-video SHA256, model, prompt, task
context, FPS and interval. Interrupted attempts consume the persisted retry/call
budget. Source changes require a new work directory. External model calls are
billable and upload selected clips to Ark; this is not local model inference.

After all episodes pass, a staging dataset is built and fully compared against
input. Linux renameat2(RENAME_NOREPLACE) publishes it atomically without replacing
an existing destination. Work contains state, caches, raw responses, token usage,
original metadata and validation proofs. Dataset directory contains no debug files.
Only verified completed output is suitable for the training upload queue. Semantic
correctness is not mathematically guaranteed by schema/coverage checks.

The disk preflight estimates 1.25 times dataset artifact size plus 256 MiB, before
any API calls. This is an estimate, not a hard upper bound; reserve additional
space for cached video clips and the conversion job. Input is never modified.

## Automatic upload

Optional targets JSON (paths must be explicitly supplied by the deployment):

```json
[
  {"name": "data-server", "destination": "/mounted-data-receiver/NEW_dataset_run001"},
  {"name": "training-server", "destination": "/mounted-training-receiver/NEW_dataset_run001"}
]
```

Pass `--upload-targets /local/config/upload-targets.json` to `run` to chain uploads,
or retry completed uploads independently:

```bash
python -m lerobot_annotation upload \
  --output /local/delivery/annotated_lerobot \
  --work /local/annotation-work/run-001 \
  --upload-targets /local/config/upload-targets.json
```

This phase supports explicitly mounted writable receiver paths. It never runs
remote commands, renames/removes remote files or overwrites existing files.
Parent receiver directories must already exist. A new dataset subdirectory is
created for this upload; this MUST be allowed by the receiver's upload policy.
If directory creation is forbidden, an existing receiver upload API is required
instead. Paths/hosts are never guessed from the sample data mount.

Every file is created exclusively and read back with SHA256 verification. Resume
skips identical files only for the same local journal. Conflicting or truncated
files block the transfer rather than being overwritten: choose a new target for
a new attempt. Data and training targets have independent journals.
meta/info.json is copied last. The receiving uploader/consumer MUST treat its
absence as incomplete; directory appearance alone is not readiness. This receiver
agreement is required before unattended production deployment. No actual remote
upload is performed by tests.

## Conversion/UI integration boundary

Do not modify the converter's immutable published generation. Let conversion
finish and verify, then invoke this command on the returned generation path.
Exit code 0 plus work/completed.json gates the upload queue. A nonzero status
is retried automatically by the job manager; conversion need not rerun.
UI is a status/control client; it must not run the model on its main thread.

The converter can invoke `/absolute/project/scripts/annotate_lerobot.py run ...`
using the annotation environment's Python. This wrapper works even when the
converter's current directory is a different repository; no PYTHONPATH change
or monkey-patching of the converter is required.

The examined `data-auto-processing` package already publishes immutable local
generations in `convest.ingest.publish.Publisher`. Its old production upload
helper replaces remote files, so it is not reused for this append-only contract.
UI wiring and actual server endpoint configuration have not been changed in
this phase. Automatic watching/scheduling of completed generations is a later
integration step, after the callable/CLI contract is accepted.

## Industrial PC container

```bash
docker build -f docker/Dockerfile.annotation -t text-label-lerobot:dev .
docker run --rm -e ARK_API_KEY -e MODEL_NAME \
  -v /local/cleaned_lerobot:/input:ro \
  -v /local/delivery:/delivery \
  -v /local/annotation-work:/work \
  text-label-lerobot:dev run --input /input \
  --output /delivery/annotated_lerobot --work /work/run-001
```

No GPU is required for this API-backed text-label phase. Full original Stage 4/5
object tracking/atomic-action export is not yet integrated into this interface.
Hardware/network/storage and model access must be verified on the actual PC.

## Validation

```bash
python -m unittest discover -s tests -p test_lerobot_roundtrip.py -v
python tests/verify_real_episode_roundtrip.py \
  --input /read-only/209_213_data_lerobot \
  --report /local/annotation-work/real_roundtrip.json
```

The real-data check uses episode_000000 with SYNTHETIC TEST LABELS to verify
non-task preservation and changes to task_index; it is not semantic annotation.
It copies that episode and three videos into a temporary test dataset, validates
an output copy, then deletes its own disposable copies. Requires roughly 1.3 GB
temporary free space. No paid model calls or remote writes occur.
