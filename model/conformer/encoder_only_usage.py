"""Example utilities for running the encoder-only Conformer."""

import argparse
from pathlib import Path
from typing import Optional

import torch

from conformer import ConformerEncoderOnly, load_encoder_from_checkpoint


def forward_pass_example(batch_size: int = 2, seq_length: int = 160, feature_dim: int = 80) -> None:
    """Run a forward pass and print resulting tensor shapes."""
    model = ConformerEncoderOnly(input_dim=feature_dim)
    inputs = torch.randn(batch_size, seq_length, feature_dim)
    input_lengths = torch.full((batch_size,), seq_length, dtype=torch.long)

    with torch.no_grad():
        hidden_states, output_lengths = model(inputs, input_lengths)

    print(f"Hidden state shape: {tuple(hidden_states.shape)}")
    print(f"Output lengths: {output_lengths.tolist()}")


def checkpoint_loading_example(checkpoint_path: str) -> None:
    """Load encoder weights from a legacy checkpoint and confirm shapes."""
    model = ConformerEncoderOnly()
    load_info = load_encoder_from_checkpoint(checkpoint_path, model)

    unexpected, missing = load_info["load_status"].unexpected_keys, load_info["load_status"].missing_keys
    print(f"Loaded encoder weights from: {checkpoint_path}")
    print(f"Skipped keys: {load_info['skipped_keys']}")
    print(f"Unexpected keys reported by PyTorch: {unexpected}")
    print(f"Missing keys reported by PyTorch: {missing}")


def main(checkpoint: Optional[str]) -> None:
    forward_pass_example()

    if checkpoint is None:
        print("No checkpoint path supplied; skipping checkpoint loading example.")
        return

    path = Path(checkpoint)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")

    checkpoint_loading_example(str(path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conformer encoder-only usage example")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional path to a legacy checkpoint")
    args = parser.parse_args()
    main(args.checkpoint)
