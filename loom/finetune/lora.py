import math

import torch
import torch.nn as nn

from config import LoRAConfig


class LoRALayer(nn.Module):
    """Low-rank update: delta_W = (alpha/rank) * B @ A, added to a frozen base layer's output.

    Why: fine-tuning all weights of even a 124M model means an optimizer state
    (AdamW = 2x params) larger than the model itself. LoRA freezes the base
    weight matrix and learns a tiny rank-r decomposition instead — cuts
    trainable params (and optimizer memory) by orders of magnitude, which
    matters directly on an 8GB card.
    """

    def __init__(self, in_dim: int, out_dim: int, rank: int, alpha: int):
        super().__init__()
        self.A = nn.Parameter(torch.randn(in_dim, rank) * (1 / math.sqrt(in_dim)))
        self.B = nn.Parameter(torch.zeros(rank, out_dim))  # zero init -> delta_W starts at 0
        self.scale = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * (x @ self.A @ self.B)


class LinearWithLoRA(nn.Module):
    """Wraps an existing nn.Linear, freezing it and adding a parallel LoRA path."""

    def __init__(self, linear: nn.Linear, rank: int, alpha: int, dropout: float = 0.0):
        super().__init__()
        self.linear = linear
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.lora(self.dropout(x))


def apply_lora(model, cfg: LoRAConfig):
    """Replace target attention projections with LoRA-wrapped versions in place.

    Why: only q/v projections get LoRA (not k) — this matches the original LoRA
    paper's finding that adapting query+value captures most of the benefit,
    keeping param count (and thus VRAM/compute) minimal.
    """
    for param in model.parameters():
        param.requires_grad = False

    for block in model.trf_blocks:
        for name in cfg.target_modules:
            original = getattr(block.att, name)
            wrapped = LinearWithLoRA(original, cfg.rank, cfg.alpha, cfg.dropout)
            setattr(block.att, name, wrapped)

    return model


def merge_lora(model):
    """Fold trained LoRA deltas into the base linear weights, restoring plain nn.Linear
    layers so the model's state_dict matches vanilla GPTModel's keys exactly.

    Why: export_to_hub.py loads checkpoints into a plain GPTModel via load_state_dict —
    a model still holding LinearWithLoRA wrappers won't match those keys. Merging lets
    a LoRA-trained model be exported/pushed with the same pipeline as any other.
    """
    for block in model.trf_blocks:
        for name in ("W_query", "W_key", "W_value"):
            module = getattr(block.att, name)
            if not isinstance(module, LinearWithLoRA):
                continue
            linear = module.linear
            delta = module.lora.scale * (module.lora.A @ module.lora.B)  # [in, out]
            linear.weight.data += delta.T
            linear.weight.requires_grad = True
            if linear.bias is not None:
                linear.bias.requires_grad = True
            setattr(block.att, name, linear)

    return model
