import torch
import torch.nn as nn

from config import GPTConfig
from loom.model.attention import MultiHeadAttention


class LayerNorm(nn.Module):
    """Custom LayerNorm (not nn.LayerNorm) so param names (scale/shift) match
    GPT-2's own checkpoint naming when loading pretrained weights — see
    loom/model/pretrained.py, which maps ln_1/ln_2/ln_f weight+bias here directly."""

    def __init__(self, emb_dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    """Tanh-approximation GELU (not torch.nn.GELU) — matches GPT-2's exact
    activation function so loaded pretrained weights produce identical outputs
    to the original model, not just "close enough"."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1 + torch.tanh(
            (2 / torch.pi) ** 0.5 * (x + 0.044715 * x ** 3)
        ))


class FeedForward(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg.emb_dim, 4 * cfg.emb_dim),
            GELU(),
            nn.Linear(4 * cfg.emb_dim, cfg.emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: norm -> attn -> residual, norm -> ffn -> residual.

    Why pre-norm (norm before sublayer) over post-norm: pre-norm keeps a clean
    gradient path through the residual stream, which is what makes deep stacks
    (12-48 layers here) trainable without careful warmup schedules."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg.emb_dim,
            d_out=cfg.emb_dim,
            context_length=cfg.context_length,
            dropout=cfg.drop_rate,
            num_heads=cfg.n_heads,
            qkv_bias=cfg.qkv_bias,
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg.emb_dim)
        self.norm2 = LayerNorm(cfg.emb_dim)
        self.drop_shortcut = nn.Dropout(cfg.drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x
