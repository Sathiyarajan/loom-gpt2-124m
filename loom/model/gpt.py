import torch
import torch.nn as nn

from config import GPTConfig
from loom.model.transformer_block import TransformerBlock, LayerNorm


class GPTModel(nn.Module):
    """Decoder-only transformer LM. tok_emb + pos_emb (learned, not sinusoidal —
    matches original GPT-2, simpler than rotary/ALiBi) -> N transformer blocks ->
    final norm -> linear head projecting back to vocab logits."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim)
        # pos_emb table size = context_length: model can't see past this length at
        # inference without retraining/interpolating position embeddings.
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.emb_dim)
        self.drop_emb = nn.Dropout(cfg.drop_rate)

        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg.n_layers)])

        self.final_norm = LayerNorm(cfg.emb_dim)
        self.out_head = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False)

    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
