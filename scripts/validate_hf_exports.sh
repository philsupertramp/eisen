#!/usr/bin/bash

og_dir=$(pwd)

(
  cd $(mktemp -d)

  cur_dir=$(pwd)

  uv init .
  uv add torch 'transformers[torch]'

  uv run ${og_dir}/scripts/validate_hf_export.py --export-dir ${og_dir}/data/hf_export_tiny_smoke
)

cd ${og_dir}
