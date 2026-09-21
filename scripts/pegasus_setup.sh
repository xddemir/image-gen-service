#!/usr/bin/env bash
#
# One-time environment setup for image-gen on the DFKI Pegasus cluster.
#
# Run this ON THE LOGIN NODE, inside the Enroot container (see scripts/README.md
# for how to get into one). The login node is the only place with internet, so
# this is where the venv is built and the weights are fetched.
#
#   ./scripts/pegasus_setup.sh                  # core only, no torch
#   ./scripts/pegasus_setup.sh --with-gpu       # + torch/diffusers
#   ./scripts/pegasus_setup.sh --with-gpu --with-weights
#
# Everything lands on /netscratch. Nothing goes in $HOME, which has a 10 GB cap
# that a single SDXL checkpoint would blow through.

set -euo pipefail

# A minimal container image does not always export USER, and `set -u` turns
# that into an unbound-variable abort before anything useful has happened.
: "${USER:=$(id -un 2>/dev/null || echo unknown)}"
export USER

NETSCRATCH="${NETSCRATCH:-/netscratch/$USER}"
VENV_DIR="${VENV_DIR:-$NETSCRATCH/venvs/image-gen}"
export HF_HOME="${HF_HOME:-$NETSCRATCH/hf}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${OUT_DIR:-$NETSCRATCH/image-gen/out}"

# Work from inside the repo and use relative paths from here on. `pip install
# -e "/abs/path[extra]"` is parsed as a requirement name rather than a path on
# some platforms, and ".[extra]" sidesteps that entirely.
cd "$REPO_DIR"

WITH_GPU=0
WITH_WEIGHTS=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --with-gpu)     WITH_GPU=1 ;;
        --with-weights) WITH_WEIGHTS=1 ;;
        # Print the header comment block, stopping at the first line that is
        # not a comment -- so this stays correct when the header is edited.
        -h|--help)
            awk 'NR==1 {next} /^#/ {sub(/^# ?/, ""); print; next} {exit}' "$0"
            exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done

say()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
warn() { printf '\033[33mwarning: %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[31merror: %s\033[0m\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- checks

say "Checking the environment"

# The single most common way to lose an afternoon here: quietly filling $HOME.
case "$VENV_DIR" in
    "$HOME"/*) die "VENV_DIR is under \$HOME ($VENV_DIR). \$HOME is capped at
       10 GB; put the venv on /netscratch instead." ;;
esac
case "$HF_HOME" in
    "$HOME"/*) die "HF_HOME is under \$HOME ($HF_HOME). SDXL alone will exceed
       the 10 GB quota. Set HF_HOME=$NETSCRATCH/hf." ;;
esac

[[ -d "$NETSCRATCH" ]] || die "$NETSCRATCH does not exist. Check \$USER, or set
       NETSCRATCH=/path/to/your/scratch."
[[ -w "$NETSCRATCH" ]] || die "$NETSCRATCH is not writable by $USER."

PYTHON="${PYTHON:-python3}"
command -v "$PYTHON" >/dev/null || die "no '$PYTHON' on PATH. Are you inside the
       container? See scripts/README.md."

PY_VERSION="$("$PYTHON" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
"$PYTHON" - <<'EOF' || die "Python >= 3.10 required (found $PY_VERSION)."
import sys
sys.exit(0 if sys.version_info >= (3, 10) else 1)
EOF

echo "  python        $PYTHON ($PY_VERSION)"
echo "  repo          $REPO_DIR"
echo "  venv          $VENV_DIR"
echo "  HF_HOME       $HF_HOME"
echo "  out           $OUT_DIR"

df -h "$NETSCRATCH" 2>/dev/null | tail -n 1 | awk '{print "  free space    " $4}'

# ------------------------------------------------------------------ venv

say "Creating the virtualenv"

mkdir -p "$(dirname "$VENV_DIR")" "$HF_HOME" "$OUT_DIR"

if [[ -d "$VENV_DIR" ]]; then
    echo "  reusing existing venv at $VENV_DIR"
else
    "$PYTHON" -m venv "$VENV_DIR"
    echo "  created $VENV_DIR"
fi

# venv layout differs: bin/ on Linux (the cluster), Scripts/ under Git Bash on
# Windows. Handling both costs nothing here and lets you smoke-test this script
# locally before pushing it to Pegasus.
if [[ -f "$VENV_DIR/bin/activate" ]]; then
    VENV_BIN="$VENV_DIR/bin"
elif [[ -f "$VENV_DIR/Scripts/activate" ]]; then
    VENV_BIN="$VENV_DIR/Scripts"
else
    die "no activate script under $VENV_DIR -- venv creation failed?"
fi

# shellcheck source=/dev/null
source "$VENV_BIN/activate"
python -m pip install --quiet --upgrade pip

# --------------------------------------------------------------- install

say "Installing image-gen"

if [[ "$WITH_GPU" -eq 1 ]]; then
    echo "  installing with the GPU extra (torch, diffusers) -- this is a few GB"
    python -m pip install -e ".[api,local]"
else
    # Deliberately torch-free: this proves the boring layer (python version,
    # deps, entry point, /netscratch writes) before torch can confuse the
    # diagnosis.
    echo "  installing core + api only (no torch)"
    python -m pip install -e ".[api]"
fi

python -c 'import image_gen; print("  image_gen", image_gen.__version__)'

if [[ "$WITH_GPU" -eq 1 ]]; then
    # Importing image_gen never pulls in torch -- backends are imported lazily
    # -- so this is the first thing that actually exercises it.
    if python -c 'import torch' >/dev/null 2>&1; then
        python - <<'EOF'
import torch
print("  torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
EOF
    else
        warn "torch installed but cannot be imported on THIS machine."
        cat <<'EOF'
  Expected on a login node: these are often VMs whose CPU model masks SSE4.2,
  so they do not meet the x86-64-v2 baseline the numpy 2.x wheels are built
  against. Nothing to fix -- installing and downloading here is fine (both are
  pure Python); run anything that imports torch on a compute node, where the
  same venv works.

  Confirm with:  grep -oE 'sse4_2|popcnt|cx16' /proc/cpuinfo | sort -u
  (empty on the login node, populated on a compute node)
EOF
    fi
fi

# --------------------------------------------------------------- weights

if [[ "$WITH_WEIGHTS" -eq 1 ]]; then
    say "Pre-fetching model weights"
    echo "  Compute nodes have no internet, so this must happen here, now."
    echo "  Expect roughly 13 GB into $HF_HOME."

    python -m pip install --quiet huggingface_hub
    image-gen download-weights --config configs/default.yaml

    echo "  revisions pinned -> $REPO_DIR/configs/revisions.lock.json"
    echo "  Commit that file: the pins are part of the reproducibility claim."
fi

# ----------------------------------------------------------- smoke test

say "Smoke test (stub backend -- no GPU, no model)"

image-gen validate examples/queue.json
image-gen run examples/queue.json --out "$OUT_DIR"
image-gen merge-manifest "$OUT_DIR"
image-gen status "$OUT_DIR"

# ------------------------------------------------------------ next steps

say "Done"

cat <<EOF
The environment works: python, deps, entry point, and writes to /netscratch.
It proves nothing yet about GPU, torch, weights or the UNet swap -- the stub
backend does not run a model.

To use this venv later (in a job script, or a new shell):

    source $VENV_BIN/activate
    export HF_HOME=$HF_HOME
    export HF_HUB_OFFLINE=1        # on compute nodes: no internet
    export TRANSFORMERS_OFFLINE=1

Next:
EOF

if [[ "$WITH_WEIGHTS" -eq 0 ]]; then
    echo "  1. Fetch the weights while you still have internet:"
    echo "       ./scripts/pegasus_setup.sh --with-gpu --with-weights"
fi

cat <<'EOF'
  2. Implement LocalDiffusersGenerator (build step 5), then verify the
     UNet-swap load path on a GPU node -- it is inferred from the repo
     layout, not documented, so first contact is the real test.
  3. Write the sbatch scripts (step 6). See scripts/README.md for the
     cluster values you need to look up first.
EOF
