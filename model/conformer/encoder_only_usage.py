"""Example utilities for running the encoder-only Conformer."""
"""
python encoder_only_usage.py \
  --checkpoint /data_x/aa007878/deep/myung/conformer/train/checkpoints/stage1_layer8_batch_256/epoch01_val1.5534.pt \
  --output /data_x/aa007878/deep/myung/model/conformer/conformer_model/conformer_stage1.pt
"""
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


def save_encoder_model(checkpoint_path: str, output_path: str, force_num_layers: Optional[int] = None) -> None:
    """Load encoder from checkpoint and save the state dict to a new file."""
    print(f"Loading from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # 1. Determine state_dict
    if "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # 2. Determine model config
    # Try to extract from checkpoint config if available
    model_cfg = checkpoint.get("config", {}).get("model", {})
    
    # Fallback or override defaults
    # Note: train_stage1_ctc.py uses "num_layers", Conformer uses "num_encoder_layers"
    if force_num_layers is not None:
        num_layers = force_num_layers
        print(f"Forcing num_layers={num_layers} (from args)")
    else:
        num_layers = model_cfg.get("num_layers", 12)
        print(f"Using num_layers={num_layers} (from config)")
    
    print(f"Initializing ConformerEncoderOnly with num_layers={num_layers}")
    
    model = ConformerEncoderOnly(
        input_dim=model_cfg.get("input_dim", 80),
        encoder_dim=model_cfg.get("encoder_dim", 512),
        num_encoder_layers=num_layers,
        num_attention_heads=model_cfg.get("num_attention_heads", 8),
        feed_forward_expansion_factor=model_cfg.get("feed_forward_expansion_factor", 4),
        conv_expansion_factor=model_cfg.get("conv_expansion_factor", 2),
        conv_kernel_size=model_cfg.get("conv_kernel_size", 31),
        input_dropout_p=model_cfg.get("dropout", 0.1),
        feed_forward_dropout_p=model_cfg.get("dropout", 0.1),
        attention_dropout_p=model_cfg.get("dropout", 0.1),
        conv_dropout_p=model_cfg.get("dropout", 0.1),
    )

    # 3. Filter and load state dict
    new_state_dict = {}
    for k, v in state_dict.items():
        # Remove DDP prefix if present
        if k.startswith("module."):
            k = k[7:]
        
        # We only want encoder parameters
        if k.startswith("encoder."):
            new_state_dict[k] = v
    
    if not new_state_dict:
        raise RuntimeError("No 'encoder.*' keys found in the checkpoint state dict!")

    keys = model.load_state_dict(new_state_dict, strict=True)
    print(f"Loaded state dict: {keys}")
    
    # Ensure output directory exists
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(model.state_dict(), out_path)
    print(f"Saved encoder state dict to: {out_path}")


def main(checkpoint: Optional[str], output: Optional[str], num_layers: Optional[int] = None) -> None:
    forward_pass_example()

    if checkpoint is None:
        print("No checkpoint path supplied; skipping checkpoint loading example.")
        return

    path = Path(checkpoint)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")

    if output:
        save_encoder_model(str(path), output, num_layers)
    else:
        checkpoint_loading_example(str(path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conformer encoder-only usage example")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional path to a legacy checkpoint")
    parser.add_argument("--output", type=str, default=None, help="Optional path to save the converted encoder model")
    parser.add_argument("--num_layers", type=int, default=None, help="Optional override for number of encoder layers")
    args = parser.parse_args()
    main(args.checkpoint, args.output, args.num_layers)
