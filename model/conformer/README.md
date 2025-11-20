# Conformer Encoder-Only Guide

This package now exposes `ConformerEncoderOnly`, a pure encoder that returns `(batch, length, hidden_dim)` representations without attaching a classifier.

## Forward pass example
```python
import torch
from conformer import ConformerEncoderOnly

model = ConformerEncoderOnly(input_dim=80, encoder_dim=512)
inputs = torch.randn(2, 320, 80)
lengths = torch.tensor([320, 280])

with torch.no_grad():
    hidden_states, hidden_lengths = model(inputs, lengths)

print(hidden_states.shape)  # -> torch.Size([2, 320, 512])
print(hidden_lengths)       # -> tensor([320, 280])
```

## Loading encoder weights from a legacy checkpoint
```python
from conformer import ConformerEncoderOnly, load_encoder_from_checkpoint

model = ConformerEncoderOnly()
load_info = load_encoder_from_checkpoint("/path/to/old.ckpt", model)
print(load_info["load_status"])  # torch.nn.modules.module._IncompatibleKeys
print(load_info["skipped_keys"]) # keys filtered out (e.g., classifier heads)
```
Keys belonging to linear heads (e.g., `fc.weight`) are ignored automatically, so you can reuse checkpoints that were trained with a classification layer.

## Quick CLI demo
Run the self-contained example script:
```bash
python encoder_only_usage.py --checkpoint /path/to/checkpoint.pt
```
Omit `--checkpoint` to only verify tensor shapes via a dummy forward pass.
