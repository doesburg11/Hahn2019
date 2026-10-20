#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"
conda env create --prefix "$repo_root/.conda" --file environment.yml
"$repo_root/.conda/bin/python" -m pip install -r requirements-cpu.txt
"$repo_root/.conda/bin/python" -m pip install -r requirements.txt --index-url https://pypi.org/simple
printf 'Activate with: conda activate %s/.conda\n' "$repo_root"
