#!/usr/bin/env bash
# Download every pretrained checkpoint required by PerformRecast from
# Google Drive. The destination directory can be overridden via the
# PERFORMRECAST_PRETRAINED environment variable; otherwise weights are
# saved under "$REPO_ROOT/pretrained_weights/".
#
# The script is idempotent: files that already exist on disk are
# skipped, so re-running it after a partial download is safe.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEST="${PERFORMRECAST_PRETRAINED:-${REPO_ROOT}/pretrained_weights}"

mkdir -p "${DEST}/performrecast"

# Always invoke gdown via `python3 -m gdown` so we use the version
# installed in the active Python environment.
GDOWN=(python3 -m gdown)

# Quick install if gdown isn't importable.
if ! python3 -c "import gdown" 2>/dev/null; then
    echo "Installing gdown into the active Python environment ..."
    python3 -m pip install -U gdown
fi

# We pass each Google Drive entry as a full ``uc?id=...`` URL, which is
# accepted by every gdown release we care about (3.x, 4.x, 5.x, 6.x).
# In gdown 6.x, fuzzy-URL parsing and the virus-scan-page bypass became
# the default behaviour, so no extra flags are needed.
_gdown_url() { echo "https://drive.google.com/uc?id=$1"; }

# --- File table -----------------------------------------------------------
# Format: "<relative-path-under-DEST>|<google-drive-file-id>|<human-name>"
FILES=(
    "performrecast/appearance_feature_extractor.pth|1YnmRePhPjtKY3Us8C0Mg-VInGVSVepB5|Appearance feature extractor (F)"
    "performrecast/motion_extractor.pth|1VhcFZrTCpJvSpSuGVSPo3bH4XsGVw5oj|Motion extractor (M)"
    "performrecast/spade_generator.pth|1G6NKloq29ULCQeJAw-4qvH0nA-5oj6-T|SPADE generator (G)"
    "performrecast/warping_module.pth|1Pvh8foijP1jHwQxnP-676UZH86krdyOK|Warping module (W)"
    "vit_base_patch16_224.dino/pytorch_model.bin|1L3HuaJNhd4fZ2E5SO2ZNxmwXpYmtXav0|DINO ViT-B/16 backbone (used inside F)"
    "landmark.onnx|1CjnT5pT1dKIE2SYuizBlhRge2fTohA4w|LivePortrait 203-pt landmark refiner"
)

# --- Download loop --------------------------------------------------------
for entry in "${FILES[@]}"; do
    rel_path="${entry%%|*}"
    rest="${entry#*|}"
    file_id="${rest%%|*}"
    human="${rest#*|}"
    abs_path="${DEST}/${rel_path}"

    if [[ -s "${abs_path}" ]]; then
        echo "[skip] ${rel_path} already exists (${human})"
        continue
    fi

    echo "[get ] ${rel_path} (${human})"
    mkdir -p "$(dirname "${abs_path}")"
    "${GDOWN[@]}" "$(_gdown_url "${file_id}")" -O "${abs_path}"
done

# --- InsightFace buffalo_l (face detector + 106-pt landmark) --------------
# Distributed as a single zip; extracted into pretrained_weights/insightface/models/.
BUFFALO_ID="1U6Tgh4Rshhr-vbedvkEr2i4J_yN-C7u9"
BUFFALO_DIR="${DEST}/insightface/models/buffalo_l"
BUFFALO_ZIP="${DEST}/insightface/models/buffalo_l.zip"
BUFFALO_SENTINEL="${BUFFALO_DIR}/det_10g.onnx"

if [[ -s "${BUFFALO_SENTINEL}" ]]; then
    echo "[skip] insightface/models/buffalo_l/ already exists"
else
    if [[ ! -s "${BUFFALO_ZIP}" ]]; then
        echo "[get ] buffalo_l.zip (InsightFace face detector + 106-pt landmark)"
        mkdir -p "$(dirname "${BUFFALO_ZIP}")"
        "${GDOWN[@]}" "$(_gdown_url "${BUFFALO_ID}")" -O "${BUFFALO_ZIP}"
    fi
    echo "[unzip] buffalo_l.zip -> $(dirname "${BUFFALO_DIR}")"

    extract_dir="$(dirname "${BUFFALO_DIR}")"
    if command -v unzip >/dev/null 2>&1; then
        if unzip -l "${BUFFALO_ZIP}" | awk '{print $NF}' | grep -q '^buffalo_l/'; then
            unzip -q -o "${BUFFALO_ZIP}" -d "${extract_dir}"
        else
            mkdir -p "${BUFFALO_DIR}"
            unzip -q -o "${BUFFALO_ZIP}" -d "${BUFFALO_DIR}"
        fi
    else
        # Fallback: use Python's zipfile module (no extra deps).
        python3 - "$BUFFALO_ZIP" "$extract_dir" "$BUFFALO_DIR" <<'PYEOF'
import sys, zipfile, os
zip_path, extract_dir, buffalo_dir = sys.argv[1], sys.argv[2], sys.argv[3]
with zipfile.ZipFile(zip_path) as z:
    nested = any(n.startswith("buffalo_l/") for n in z.namelist())
    target = extract_dir if nested else buffalo_dir
    os.makedirs(target, exist_ok=True)
    z.extractall(target)
PYEOF
    fi
    rm -f "${BUFFALO_ZIP}"
fi

# --- Verification ---------------------------------------------------------
echo
echo "Verifying download ..."
missing=0
for entry in "${FILES[@]}"; do
    rel_path="${entry%%|*}"
    if [[ ! -s "${DEST}/${rel_path}" ]]; then
        echo "[MISSING] ${DEST}/${rel_path}"
        missing=$((missing + 1))
    fi
done

if [[ ! -s "${BUFFALO_SENTINEL}" ]]; then
    echo "[MISSING] ${BUFFALO_SENTINEL}"
    missing=$((missing + 1))
fi

if [[ ${missing} -gt 0 ]]; then
    echo
    echo "${missing} file(s) failed to download. Re-run this script to retry." >&2
    exit 1
fi

echo
echo "All weights downloaded to:"
echo "  ${DEST}"
echo
echo "Layout:"
( cd "${DEST}" && find . -maxdepth 3 -type f | sort | sed 's|^\./|  |' )
echo
echo "(FLAME_masks.pkl ships in this repo's assets/FLAME_masks/ directory"
echo " and is loaded automatically.)"
