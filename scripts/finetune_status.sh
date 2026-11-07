#!/bin/sh
# Read-only status; works from any directory without activating the environment.
set -eu
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec "$script_dir/../.venv/bin/python" "$script_dir/finetune_status.py" "$@"
