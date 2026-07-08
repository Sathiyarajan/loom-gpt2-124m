from transformers import PretrainedConfig


class LoomConfig(PretrainedConfig):
    """HF-compatible config wrapping our GPTConfig fields (see ../config.py)."""

    model_type = "loom_gpt"

    def __init__(
        self,
        vocab_size: int = 50257,
        context_length: int = 1024,
        emb_dim: int = 768,
        n_heads: int = 12,
        n_layers: int = 12,
        drop_rate: float = 0.1,
        qkv_bias: bool = True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.drop_rate = drop_rate
        self.qkv_bias = qkv_bias
        super().__init__(**kwargs)
