#!/usr/bin/env bash
# Download and verify model checkpoints that geo-eval-harness does not fetch on its own, so every
# hub has byte-identical copies.
#
# Usage:
#   prep_hub_assets.sh                 leader: download, then write the lock file
#   prep_hub_assets.sh --lock FILE     other hubs: download exactly the locked revisions, verify sha256
#   prep_hub_assets.sh --check [--lock FILE]   verify only, no downloads
#
# Environment (defaults in brackets):
#   GEH_DIR      [~/geo-eval-harness]     repo checkout with .venv/
#   CKPT_DIR     [$GEH_DIR/checkpoints]   TerraFM-B.pth goes here
#   HF_HOME      [~/.cache/huggingface]   Hugging Face cache (UniverSat weights)
#   TORCH_HOME   [~/.cache/torch]         torch.hub cache (UniverSat code)
# The lock file defaults to $CKPT_DIR/assets.lock. Copy it to the other hubs.
set -euo pipefail

GEH_DIR=${GEH_DIR:-$HOME/geo-eval-harness}
CKPT_DIR=${CKPT_DIR:-$GEH_DIR/checkpoints}
PYTHON=${PYTHON:-$GEH_DIR/.venv/bin/python}
LOCK=$CKPT_DIR/assets.lock
MODE=download
HAVE_LOCK=0
while (($#)); do
  case $1 in
    --lock) LOCK=$(realpath "$2"); HAVE_LOCK=1; shift ;;
    --check) MODE=check ;;
    -h|--help) sed -n 2,16p "$0"; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
  shift
done
[[ -f $LOCK ]] && HAVE_LOCK=1

TERRAFM_REPO=MBZUAI/TerraFM
TERRAFM_FILE=TerraFM-B.pth
UNIVERSAT_HUB=gastruc/UniverSat
UNIVERSAT_REF=d4e8712d29651c34c4fdb9f9d2c0b168dd5c678b  # pinned in geo_eval_harness/models/universat.py
UNIVERSAT_HF=g-astruc/UniverSat

fail=0
say() { printf '%-8s %s\n' "$1" "$2"; }
locked() { [[ $HAVE_LOCK == 1 ]] && awk -v k="$1" '$1 == k {print $2}' "$LOCK"; }
[[ -x $PYTHON ]] || { echo "python not found: $PYTHON (set GEH_DIR or PYTHON)" >&2; exit 2; }
mkdir -p "$CKPT_DIR"
echo "hub=$(cat /etc/hostname) geh=$GEH_DIR ckpt=$CKPT_DIR lock=$LOCK ($([[ $HAVE_LOCK == 1 ]] && echo present || echo none)) mode=$MODE"

# --- TerraFM: one file from Hugging Face at a pinned commit ---------------------------------
terrafm_rev=$(locked terrafm_revision || true)
if [[ -z $terrafm_rev && $MODE == download ]]; then
  terrafm_rev=$(curl -sSL --fail "https://huggingface.co/api/models/$TERRAFM_REPO/revision/main" \
    | "$PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["sha"])')
fi
target=$CKPT_DIR/$TERRAFM_FILE
if [[ ! -f $target && $MODE == download ]]; then
  echo "downloading $TERRAFM_REPO@$terrafm_rev/$TERRAFM_FILE"
  curl -SL --fail -C - -o "$target.part" \
    "https://huggingface.co/$TERRAFM_REPO/resolve/$terrafm_rev/$TERRAFM_FILE"
  mv "$target.part" "$target"
fi
if [[ -f $target ]]; then
  terrafm_sha=$(sha256sum "$target" | cut -d' ' -f1)
  want=$(locked terrafm_sha256 || true)
  if [[ -n $want && $want != "$terrafm_sha" ]]; then say MISMATCH "$target sha256 $terrafm_sha != locked $want"; fail=1
  else say OK "$target ($(du -h "$target" | cut -f1)) sha256=${terrafm_sha:0:16}"; fi
else
  say MISSING "$target"; fail=1
fi

# --- UniverSat: torch.hub code at the pinned commit + Hugging Face weights ------------------
universat_hf_rev=$(locked universat_hf_revision || true)
if [[ $MODE == download ]]; then
  echo "loading UniverSat ($UNIVERSAT_HUB@$UNIVERSAT_REF, weights ${universat_hf_rev:-latest})"
  UNIVERSAT_HF_REV=$universat_hf_rev "$PYTHON" - <<PY
import os, torch
rev = os.environ.get("UNIVERSAT_HF_REV")
if rev:
    from huggingface_hub import snapshot_download
    snapshot_download("$UNIVERSAT_HF", revision=rev,
                      allow_patterns=["config.json", "*.safetensors", "*.bin", "*.pt", "*.pth"])
torch.hub.load("$UNIVERSAT_HUB:$UNIVERSAT_REF", "from_pretrained", trust_repo=True,
               skip_validation=True, compile=False)
print("UniverSat loaded")
PY
fi
universat_state=$("$PYTHON" - <<'PY'
import hashlib, json
from pathlib import Path
try:
    from huggingface_hub import scan_cache_dir
    repos = [r for r in scan_cache_dir().repos if r.repo_id == "g-astruc/UniverSat"]
except Exception:  # cache missing or unreadable
    repos = []
if not repos:
    print(json.dumps({}))
else:
    rev = max(repos[0].revisions, key=lambda r: r.last_modified)
    digest = hashlib.sha256()
    # Only files the model loads; README/.gitattributes may or may not be cached.
    used = [f for f in rev.files
            if f.file_name == "config.json" or f.file_name.endswith((".safetensors", ".bin", ".pt", ".pth"))]
    for f in sorted(used, key=lambda f: f.file_name):
        h = hashlib.sha256(Path(f.file_path).read_bytes()).hexdigest()
        digest.update(f"{f.file_name} {h}\n".encode())
    print(json.dumps({"revision": rev.commit_hash, "sha256": digest.hexdigest()}))
PY
)
universat_rev=$(echo "$universat_state" | "$PYTHON" -c 'import json,sys; print(json.load(sys.stdin).get("revision",""))')
universat_sha=$(echo "$universat_state" | "$PYTHON" -c 'import json,sys; print(json.load(sys.stdin).get("sha256",""))')
hub_dir=$(ls -d "${TORCH_HOME:-$HOME/.cache/torch}"/hub/gastruc_UniverSat_* 2>/dev/null | head -1 || true)
if [[ -z $universat_sha || -z $hub_dir ]]; then
  say MISSING "UniverSat (hub code: ${hub_dir:-none}, HF weights: ${universat_rev:-none})"; fail=1
else
  want=$(locked universat_hf_sha256 || true)
  if [[ -n $want && $want != "$universat_sha" ]]; then say MISMATCH "UniverSat weights $universat_sha != locked $want"; fail=1
  else say OK "UniverSat code $hub_dir, weights rev ${universat_rev:0:12} sha256=${universat_sha:0:16}"; fi
fi

# --- Lock file: written once on the leader, then copied to the other hubs -------------------
if [[ $MODE == download && $HAVE_LOCK == 0 && $fail == 0 ]]; then
  cat >"$LOCK" <<EOF
# geo-eval-harness hub assets lock, written $(date -u +%FT%TZ) on $(cat /etc/hostname)
terrafm_revision $terrafm_rev
terrafm_sha256 $terrafm_sha
universat_hub_ref $UNIVERSAT_REF
universat_hf_revision $universat_rev
universat_hf_sha256 $universat_sha
EOF
  echo "wrote $LOCK; copy it to the other hubs and run: $0 --lock assets.lock"
fi
((fail == 0)) && echo "all assets OK" || echo "PROBLEMS found (see above)"
exit "$fail"
