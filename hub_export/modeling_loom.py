"""Self-contained HF model file.

Deliberately duplicates loom/model/{attention,transformer_block,gpt}.py rather than
importing them: this file is uploaded standalone to the Hub, and anyone loading it via
trust_remote_code only receives the files present in the Hub repo, not the rest of this
GitHub project. Keep the two copies in sync manually if the architecture changes — the
canonical/maintained version lives in loom/model/, this is a frozen export of it.
"""
import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput

from configuration_loom import LoomConfig


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1).bool())

    def forward(self, x):
        b, num_tokens, _ = x.shape
        q = self.W_query(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.W_key(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.W_value(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        attn_scores = q @ k.transpose(-2, -1)
        attn_scores.masked_fill_(self.mask[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = (attn_weights @ v).transpose(1, 2).contiguous().view(b, num_tokens, self.d_out)
        return self.out_proj(context)


class LayerNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return self.scale * (x - mean) / torch.sqrt(var + self.eps) + self.shift


class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh((2 / torch.pi) ** 0.5 * (x + 0.044715 * x**3)))


class FeedForward(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(emb_dim, 4 * emb_dim), GELU(), nn.Linear(4 * emb_dim, emb_dim))

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: LoomConfig):
        super().__init__()
        self.att = MultiHeadAttention(cfg.emb_dim, cfg.emb_dim, cfg.context_length, cfg.drop_rate, cfg.n_heads, cfg.qkv_bias)
        self.ff = FeedForward(cfg.emb_dim)
        self.norm1 = LayerNorm(cfg.emb_dim)
        self.norm2 = LayerNorm(cfg.emb_dim)
        self.drop_shortcut = nn.Dropout(cfg.drop_rate)

    def forward(self, x):
        shortcut = x
        x = self.att(self.norm1(x))
        x = self.drop_shortcut(x) + shortcut
        shortcut = x
        x = self.ff(self.norm2(x))
        x = self.drop_shortcut(x) + shortcut
        return x


class LoomGPTForCausalLM(PreTrainedModel):
    """HF-loadable version of loom/model/gpt.py's GPTModel — see module docstring."""

    config_class = LoomConfig

    def __init__(self, config: LoomConfig):
        super().__init__(config)
        self.tok_emb = nn.Embedding(config.vocab_size, config.emb_dim)
        self.pos_emb = nn.Embedding(config.context_length, config.emb_dim)
        self.drop_emb = nn.Dropout(config.drop_rate)
        self.trf_blocks = nn.Sequential(*[TransformerBlock(config) for _ in range(config.n_layers)])
        self.final_norm = LayerNorm(config.emb_dim)
        self.out_head = nn.Linear(config.emb_dim, config.vocab_size, bias=False)

    def forward(self, input_ids, labels=None, **kwargs):
        b, seq_len = input_ids.shape
        x = self.tok_emb(input_ids) + self.pos_emb(torch.arange(seq_len, device=input_ids.device))
        x = self.trf_blocks(self.drop_emb(x))
        logits = self.out_head(self.final_norm(x))

        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), labels.flatten())
        return CausalLMOutput(loss=loss, logits=logits)

    @torch.no_grad()
    def generate_simple(self, input_ids, max_new_tokens: int, temperature: float = 0.0, top_k: int | None = None):
        """Minimal greedy/top-k/temperature sampler — not HF's full .generate() API
        (no beam search, no KV cache), kept simple since this is an educational model."""
        idx = input_ids
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.context_length :]
            logits = self(idx_cond).logits[:, -1, :]

            if top_k is not None:
                top_logits, _ = torch.topk(logits, top_k)
                logits = torch.where(logits < top_logits[:, -1:], torch.tensor(-torch.inf).to(logits.device), logits)

            if temperature > 0.0:
                probs = torch.softmax(logits / temperature, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)

            idx = torch.cat((idx, idx_next), dim=1)
        return idx
