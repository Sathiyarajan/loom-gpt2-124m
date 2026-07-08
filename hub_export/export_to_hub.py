"""Convert our trained GPTModel checkpoint into the HF-loadable LoomGPTForCausalLM,
verify output parity against the original, then save + (optionally) push to Hub.

Usage (run as a module from repo root, so relative imports in modeling_loom.py
resolve the same way they will when HF loads this package via trust_remote_code):
    cd ~/projects/loom
    PYTHONPATH=. python -m hub_export.export_to_hub \
        --checkpoint checkpoints/demo/checkpoint.pt --repo-id YOUR_USERNAME/loom-llm
"""
import argparse
import os

import torch

from config import GPT_CONFIG_124M_PRETRAINED
from loom.model.gpt import GPTModel

from .configuration_loom import LoomConfig
from .modeling_loom import LoomGPTForCausalLM


def build_hf_model(gpt_cfg, state_dict) -> LoomGPTForCausalLM:
    hf_config = LoomConfig(
        vocab_size=gpt_cfg.vocab_size,
        context_length=gpt_cfg.context_length,
        emb_dim=gpt_cfg.emb_dim,
        n_heads=gpt_cfg.n_heads,
        n_layers=gpt_cfg.n_layers,
        drop_rate=gpt_cfg.drop_rate,
        qkv_bias=gpt_cfg.qkv_bias,
    )
    # registers this repo's custom code so `trust_remote_code=True` knows what to fetch;
    # save_pretrained/push_to_hub auto-copy configuration_loom.py + modeling_loom.py
    # alongside the weights because of this mapping.
    hf_config.auto_map = {
        "AutoConfig": "configuration_loom.LoomConfig",
        "AutoModelForCausalLM": "modeling_loom.LoomGPTForCausalLM",
    }
    hf_model = LoomGPTForCausalLM(hf_config)

    # our GPTModel's own state_dict keys (tok_emb, pos_emb, trf_blocks.*, final_norm, out_head)
    # match this wrapper's top-level attribute names 1:1 -- direct load, no remapping needed.
    hf_model.load_state_dict(state_dict)
    return hf_model


def verify_parity(original_model, hf_model, gpt_cfg, device):
    original_model.to(device).eval()
    hf_model.to(device).eval()
    dummy = torch.randint(0, gpt_cfg.vocab_size, (1, 16), device=device)
    with torch.no_grad():
        original_logits = original_model(dummy)
        hf_logits = hf_model(dummy).logits
    max_diff = (original_logits - hf_logits).abs().max().item()
    print(f"max logit diff between original and HF-wrapped model: {max_diff:.2e}")
    assert max_diff < 1e-4, "parity check failed — HF wrapper does not match original model"
    print("parity check passed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--repo-id", required=True, help="e.g. your-username/loom-llm")
    parser.add_argument("--private", action="store_true", default=True)
    parser.add_argument("--push", action="store_true", help="actually push; omit to just save locally + verify")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpt_cfg = GPT_CONFIG_124M_PRETRAINED

    original_model = GPTModel(gpt_cfg)
    optimizer = torch.optim.AdamW(original_model.parameters())
    checkpoint = torch.load(args.checkpoint, map_location=device)
    original_model.load_state_dict(checkpoint["model_state_dict"])

    hf_model = build_hf_model(gpt_cfg, checkpoint["model_state_dict"])
    verify_parity(original_model, hf_model, gpt_cfg, device)

    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "export")
    hf_model.save_pretrained(save_dir)
    print(f"saved locally to {save_dir}/ (config.json + model.safetensors)")

    if args.push:
        hf_model.push_to_hub(args.repo_id, private=args.private)
        print(f"pushed to https://huggingface.co/{args.repo_id}")
        print("NOTE: also copy configuration_loom.py and modeling_loom.py into the repo")
        print("      (huggingface-cli upload or web UI) so trust_remote_code works for others.")
    else:
        print("--push not set: verified + saved locally only, nothing sent to the Hub")


if __name__ == "__main__":
    main()
