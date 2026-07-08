from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int
    context_length: int
    emb_dim: int
    n_heads: int
    n_layers: int
    drop_rate: float = 0.1
    qkv_bias: bool = False


@dataclass
class TrainConfig:
    batch_size: int = 4
    num_epochs: int = 10
    learning_rate: float = 5e-4
    weight_decay: float = 0.1
    warmup_steps: int = 100
    eval_freq: int = 500
    eval_iter: int = 5
    grad_clip: float = 1.0
    device: str = "cuda"
    checkpoint_dir: str = "checkpoints"


@dataclass
class LoRAConfig:
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: tuple = ("W_query", "W_value")  # attention proj names as defined in loom/model/attention.py


GPT_CONFIG_124M = GPTConfig(vocab_size=50257, context_length=256, emb_dim=768, n_heads=12, n_layers=12)
GPT_CONFIG_355M = GPTConfig(vocab_size=50257, context_length=1024, emb_dim=1024, n_heads=16, n_layers=24)
GPT_CONFIG_774M = GPTConfig(vocab_size=50257, context_length=1024, emb_dim=1280, n_heads=20, n_layers=36)
GPT_CONFIG_1558M = GPTConfig(vocab_size=50257, context_length=1024, emb_dim=1600, n_heads=25, n_layers=48)

# Pretrained-compatible variant: context_length=1024 (matches GPT-2 pos_emb table),
# qkv_bias=True (GPT-2 attention layers use bias).
GPT_CONFIG_124M_PRETRAINED = GPTConfig(vocab_size=50257, context_length=1024, emb_dim=768, n_heads=12, n_layers=12, qkv_bias=True)
GPT_CONFIG_355M_PRETRAINED = GPTConfig(vocab_size=50257, context_length=1024, emb_dim=1024, n_heads=16, n_layers=24, qkv_bias=True)

ACTIVE_MODEL_CONFIG = GPT_CONFIG_124M
ACTIVE_TRAIN_CONFIG = TrainConfig()
ACTIVE_LORA_CONFIG = LoRAConfig()
