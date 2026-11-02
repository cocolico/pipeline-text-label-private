#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
API_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
EGO_ROOT="$(cd "${API_ROOT}/.." && pwd)"
VENV_PATH="${1:-${API_ROOT}/.venv_bbox_tracking}"
WILOR_PYTHON="${WILOR_PYTHON:-${HOME}/miniconda3/envs/wilor/bin/python}"
SAM3_PYTHON="${SAM3_PYTHON:-${HOME}/miniconda3/envs/sam3/bin/python}"
SAM3_REPO="${SAM3_REPO:-${HOME}/caoheng/sam3}"
SAM3_SOURCE_CHECKPOINT="${SAM3_SOURCE_CHECKPOINT:-${SAM3_REPO}/ckpt/sam3.pt}"
SAM3_BPE_SOURCE="${SAM3_BPE_SOURCE:-${SAM3_REPO}/sam3/assets/bpe_simple_vocab_16e6.txt.gz}"
HAND_DETECTOR_SOURCE="${HAND_DETECTOR_SOURCE:-${EGO_ROOT}/hand_pipeline/local/pretrained_models/detector.pt}"
MODEL_DIR="${API_ROOT}/models/sam3"
MODEL_LINK="${MODEL_DIR}/sam3.pt"
BPE_LINK="${MODEL_DIR}/bpe_simple_vocab_16e6.txt.gz"
HAND_MODEL_DIR="${API_ROOT}/models/wilor"
HAND_DETECTOR_LINK="${HAND_MODEL_DIR}/detector.pt"

for path in \
  "${WILOR_PYTHON}" \
  "${SAM3_PYTHON}" \
  "${SAM3_REPO}/sam3/model_builder.py" \
  "${SAM3_BPE_SOURCE}" \
  "${SAM3_SOURCE_CHECKPOINT}" \
  "${HAND_DETECTOR_SOURCE}"; do
  if [[ ! -f "${path}" ]]; then
    printf 'Missing required Stage 5 file: %s\n' "${path}" >&2
    exit 1
  fi
done

if [[ ! -x "${VENV_PATH}/bin/python" ]]; then
  "${WILOR_PYTHON}" -m venv --system-site-packages "${VENV_PATH}"
fi
"${VENV_PATH}/bin/python" -m pip install \
  --disable-pip-version-check \
  --requirement "${SCRIPT_DIR}/requirements_stage5_sam3.txt"

mkdir -p "${MODEL_DIR}" "${HAND_MODEL_DIR}"
ln -sfn "${SAM3_SOURCE_CHECKPOINT}" "${MODEL_LINK}"
ln -sfn "${SAM3_BPE_SOURCE}" "${BPE_LINK}"
ln -sfn "${HAND_DETECTOR_SOURCE}" "${HAND_DETECTOR_LINK}"
CHECKPOINT_SHA256="$(sha256sum "${SAM3_SOURCE_CHECKPOINT}" | awk '{print $1}')"
BPE_SHA256="$(sha256sum "${SAM3_BPE_SOURCE}" | awk '{print $1}')"
HAND_DETECTOR_SHA256="$(sha256sum "${HAND_DETECTOR_SOURCE}" | awk '{print $1}')"
printf '%s  %s\n' "${CHECKPOINT_SHA256}" "$(basename "${SAM3_SOURCE_CHECKPOINT}")" \
  > "${SAM3_SOURCE_CHECKPOINT}.sha256"
SAM3_COMMIT="$(git -C "${SAM3_REPO}" rev-parse HEAD 2>/dev/null || true)"

SAM3_REPO="${SAM3_REPO}" \
SAM3_PYTHON="${SAM3_PYTHON}" \
SAM3_SOURCE_CHECKPOINT="${SAM3_SOURCE_CHECKPOINT}" \
MODEL_LINK="${MODEL_LINK}" \
BPE_LINK="${BPE_LINK}" \
HAND_DETECTOR_LINK="${HAND_DETECTOR_LINK}" \
CHECKPOINT_SHA256="${CHECKPOINT_SHA256}" \
BPE_SHA256="${BPE_SHA256}" \
HAND_DETECTOR_SHA256="${HAND_DETECTOR_SHA256}" \
SAM3_COMMIT="${SAM3_COMMIT}" \
"${VENV_PATH}/bin/python" - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

link = Path(os.environ["MODEL_LINK"])
bpe = Path(os.environ["BPE_LINK"])
hand_detector = Path(os.environ["HAND_DETECTOR_LINK"])
payload = {
    "model": "SAM3",
    "checkpoint_link": str(link),
    "checkpoint_resolved": str(link.resolve()),
    "checkpoint_size_bytes": link.stat().st_size,
    "checkpoint_sha256": os.environ["CHECKPOINT_SHA256"],
    "bpe_link": str(bpe),
    "bpe_resolved": str(bpe.resolve()),
    "bpe_size_bytes": bpe.stat().st_size,
    "bpe_sha256": os.environ["BPE_SHA256"],
    "hand_detector_link": str(hand_detector),
    "hand_detector_resolved": str(hand_detector.resolve()),
    "hand_detector_size_bytes": hand_detector.stat().st_size,
    "hand_detector_sha256": os.environ["HAND_DETECTOR_SHA256"],
    "sam3_repo": os.environ["SAM3_REPO"],
    "sam3_repo_commit": os.environ["SAM3_COMMIT"],
    "sam3_python": os.environ["SAM3_PYTHON"],
    "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
}
(link.parent / "MODEL_MANIFEST.json").write_text(
    json.dumps(payload, indent=2) + "\n", encoding="utf-8"
)
PY

"${VENV_PATH}/bin/python" - <<'PY'
import cv2
import openai
import torch
import ultralytics

print("Stage 5 parent environment OK")
print("torch:", torch.__version__)
print("CUDA available:", bool(torch.cuda.is_available()))
PY

SAM3_REPO="${SAM3_REPO}" SAM3_CHECKPOINT="${MODEL_LINK}" "${SAM3_PYTHON}" - <<'PY'
import os
import sys
from pathlib import Path

repo = Path(os.environ["SAM3_REPO"])
sys.path.insert(0, str(repo))
import torch
from sam3.model_builder import build_sam3_video_predictor

checkpoint = Path(os.environ["SAM3_CHECKPOINT"])
assert torch.cuda.is_available(), "SAM3 requires a CUDA GPU on this server"
assert checkpoint.is_file(), checkpoint
assert callable(build_sam3_video_predictor)
print("SAM3 runtime OK")
print("torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("checkpoint:", checkpoint.resolve(), checkpoint.stat().st_size)
PY

printf 'Stage 5 Python: %s/bin/python\n' "${VENV_PATH}"
printf 'SAM3 Python: %s\n' "${SAM3_PYTHON}"
printf 'SAM3 checkpoint: %s -> %s\n' "${MODEL_LINK}" "${SAM3_SOURCE_CHECKPOINT}"
printf 'SAM3 SHA-256: %s\n' "${CHECKPOINT_SHA256}"
printf 'SAM3 BPE: %s -> %s\n' "${BPE_LINK}" "${SAM3_BPE_SOURCE}"
printf 'Hand detector: %s -> %s\n' "${HAND_DETECTOR_LINK}" "${HAND_DETECTOR_SOURCE}"
