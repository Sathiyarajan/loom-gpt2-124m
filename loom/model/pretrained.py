import torch

from config import GPTConfig
from loom.model.gpt import GPTModel

_HF_NAMES = {
    "gpt2-small (124M)": "gpt2",
    "gpt2-medium (355M)": "gpt2-medium",
}


def load_pretrained_gpt2(cfg: GPTConfig, model_name: str = "gpt2-small (124M)") -> GPTModel:
    """Load OpenAI GPT-2 weights (via HF hub) into our GPTModel architecture.

    cfg must have context_length=1024 and qkv_bias=True to match GPT-2's shapes
    (see GPT_CONFIG_124M_PRETRAINED / GPT_CONFIG_355M_PRETRAINED in config.py).
    """
    from transformers import GPT2Model

    hf_model = GPT2Model.from_pretrained(_HF_NAMES[model_name])
    hf_sd = hf_model.state_dict()

    model = GPTModel(cfg)
    sd = model.state_dict()

    sd["tok_emb.weight"] = hf_sd["wte.weight"]
    sd["pos_emb.weight"] = hf_sd["wpe.weight"]

    for i in range(cfg.n_layers):
        prefix = f"h.{i}."
        q_w, k_w, v_w = hf_sd[prefix + "attn.c_attn.weight"].chunk(3, dim=-1)
        q_b, k_b, v_b = hf_sd[prefix + "attn.c_attn.bias"].chunk(3, dim=-1)
        sd[f"trf_blocks.{i}.att.W_query.weight"] = q_w.T
        sd[f"trf_blocks.{i}.att.W_key.weight"] = k_w.T
        sd[f"trf_blocks.{i}.att.W_value.weight"] = v_w.T
        sd[f"trf_blocks.{i}.att.W_query.bias"] = q_b
        sd[f"trf_blocks.{i}.att.W_key.bias"] = k_b
        sd[f"trf_blocks.{i}.att.W_value.bias"] = v_b

        sd[f"trf_blocks.{i}.att.out_proj.weight"] = hf_sd[prefix + "attn.c_proj.weight"].T
        sd[f"trf_blocks.{i}.att.out_proj.bias"] = hf_sd[prefix + "attn.c_proj.bias"]

        sd[f"trf_blocks.{i}.ff.layers.0.weight"] = hf_sd[prefix + "mlp.c_fc.weight"].T
        sd[f"trf_blocks.{i}.ff.layers.0.bias"] = hf_sd[prefix + "mlp.c_fc.bias"]
        sd[f"trf_blocks.{i}.ff.layers.2.weight"] = hf_sd[prefix + "mlp.c_proj.weight"].T
        sd[f"trf_blocks.{i}.ff.layers.2.bias"] = hf_sd[prefix + "mlp.c_proj.bias"]

        sd[f"trf_blocks.{i}.norm1.scale"] = hf_sd[prefix + "ln_1.weight"]
        sd[f"trf_blocks.{i}.norm1.shift"] = hf_sd[prefix + "ln_1.bias"]
        sd[f"trf_blocks.{i}.norm2.scale"] = hf_sd[prefix + "ln_2.weight"]
        sd[f"trf_blocks.{i}.norm2.shift"] = hf_sd[prefix + "ln_2.bias"]

    sd["final_norm.scale"] = hf_sd["ln_f.weight"]
    sd["final_norm.shift"] = hf_sd["ln_f.bias"]
    sd["out_head.weight"] = hf_sd["wte.weight"]  # weight tying, as in original GPT-2

    model.load_state_dict(sd)
    return model
