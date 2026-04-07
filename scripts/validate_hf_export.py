#!/usr/bin/env python3
"""
Validate an Eisen HF export bundle with both:
  1) PyTorch + safetensors (direct tensor load)
  2) Hugging Face transformers (LlamaForCausalLM.from_pretrained)

Usage:
  python scripts/validate_hf_export.py --export-dir data/hf_export
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


def _require_deps() -> tuple[object, object, object]:
    try:
        import torch  # type: ignore
        from safetensors.torch import load_file  # type: ignore
        from transformers import LlamaForCausalLM  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "Missing dependencies. Install with:\n"
            "  pip install torch safetensors transformers\n"
            f"Original import error: {exc}"
        )
    return torch, load_file, LlamaForCausalLM


def _assert_files(export_dir: Path) -> tuple[Path, Path]:
    model_path = export_dir / "model.safetensors"
    config_path = export_dir / "config.json"
    if not model_path.exists():
        raise SystemExit(f"Missing file: {model_path}")
    if not config_path.exists():
        raise SystemExit(f"Missing file: {config_path}")
    return model_path, config_path


def _print_section(title: str) -> None:
    print(f"\n=== {title} ===")


def _validate_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    required = [
        "model_type",
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
    ]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise SystemExit(f"config.json missing keys: {missing}")
    print(f"Config OK. model_type={cfg['model_type']} hidden={cfg['hidden_size']}")
    return cfg


def _validate_safetensors(load_file, model_path: Path) -> dict:
    state = load_file(str(model_path))
    if not state:
        raise SystemExit("model.safetensors contains no tensors.")
    total_params = sum(v.numel() for v in state.values())
    print(f"Loaded {len(state)} tensors with {total_params:,} scalar params from safetensors.")
    return state


def _sample_keys(keys: Iterable[str], n: int = 6) -> list[str]:
    out = sorted(keys)
    return out[:n]


def _validate_transformers(torch, LlamaForCausalLM, export_dir: Path, cfg: dict, state: dict) -> None:
    model = LlamaForCausalLM.from_pretrained(
        str(export_dir),
        local_files_only=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=False,
    )
    model.eval()

    print("Transformers load OK.")
    print(f"Model parameters in instantiated HF module: {sum(p.numel() for p in model.parameters()):,}")

    # Verify a representative subset of keys was loaded exactly.
    model_state = model.state_dict()
    common = [k for k in state.keys() if k in model_state]
    if not common:
        raise SystemExit("No overlapping tensor keys between safetensors and HF model state_dict.")

    probe = _sample_keys(common, n=min(8, len(common)))
    mismatched = []
    for k in probe:
        if model_state[k].shape != state[k].shape:
            mismatched.append((k, tuple(state[k].shape), tuple(model_state[k].shape)))
            continue
        # quick equality check
        if not torch.allclose(model_state[k].cpu(), state[k].cpu(), atol=0.0, rtol=0.0):
            mismatched.append((k, "value_mismatch", "value_mismatch"))
    if mismatched:
        raise SystemExit(f"Loaded model tensors differ from safetensors for sample keys: {mismatched}")

    # Smoke-test a forward pass
    seq_len = min(8, int(cfg.get("max_position_embeddings", 8)))
    vocab = int(cfg["vocab_size"])
    input_ids = torch.randint(low=0, high=vocab, size=(1, seq_len), dtype=torch.long)
    with torch.no_grad():
        out = model(input_ids=input_ids)
    print(f"Forward pass OK. logits shape={tuple(out.logits.shape)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Eisen HF export bundle.")
    parser.add_argument("--export-dir", type=Path, required=True, help="Directory with model.safetensors and config.json")
    args = parser.parse_args()

    torch, load_file, LlamaForCausalLM = _require_deps()
    model_path, config_path = _assert_files(args.export_dir)

    _print_section("Config")
    cfg = _validate_config(config_path)

    _print_section("Safetensors (PyTorch)")
    state = _validate_safetensors(load_file, model_path)

    _print_section("Transformers")
    _validate_transformers(torch, LlamaForCausalLM, args.export_dir, cfg, state)

    _print_section("Result")
    print("HF export validation passed.")


if __name__ == "__main__":
    main()
