"""Compare base pretrained GPT-2 vs a fine-tuned checkpoint on the same prompts."""
import argparse

import torch

from config import GPT_CONFIG_124M_PRETRAINED
from loom.eval.metrics import generate_text
from loom.model.pretrained import load_pretrained_gpt2
from loom.tokenizer import Tokenizer
from loom.train.checkpoint import load_checkpoint

SAMPLE_PROMPTS = [
    "Every effort moves you",
    "The best way to learn programming is",
]


def run(checkpoint_path: str | None, max_new_tokens: int = 40):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = GPT_CONFIG_124M_PRETRAINED
    tok = Tokenizer()

    base_model = load_pretrained_gpt2(cfg, "gpt2-small (124M)")
    base_model.to(device).eval()

    fine_tuned = None
    if checkpoint_path:
        fine_tuned = load_pretrained_gpt2(cfg, "gpt2-small (124M)")
        optimizer = torch.optim.AdamW(fine_tuned.parameters())
        load_checkpoint(fine_tuned, optimizer, checkpoint_path, device=device)
        fine_tuned.to(device).eval()

    for prompt in SAMPLE_PROMPTS:
        ids = torch.tensor(tok.encode(prompt)).unsqueeze(0).to(device)
        base_out = generate_text(base_model, ids, max_new_tokens, cfg.context_length)
        print(f"[BASE]       {tok.decode(base_out.squeeze(0).tolist())}")

        if fine_tuned is not None:
            ft_out = generate_text(fine_tuned, ids, max_new_tokens, cfg.context_length)
            print(f"[FINE-TUNED] {tok.decode(ft_out.squeeze(0).tolist())}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=40)
    args = parser.parse_args()
    run(args.checkpoint, args.max_new_tokens)
