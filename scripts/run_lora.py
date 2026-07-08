"""LoRA instruction fine-tuning CLI — parameter-efficient alternative to
run_finetune_instruction.py. See README onboarding step 10 and the "Publishing" section
(LoRA instruction-tuned) for background. Merges adapters back into the base weights
before saving, so the checkpoint loads with a plain GPTModel like any other."""
import argparse
import functools

import torch
from torch.utils.data import DataLoader, Subset

from config import GPT_CONFIG_124M_PRETRAINED, LoRAConfig
from loom.finetune.instruction import InstructionDataset, instruction_collate_fn, train_instruction_model
from loom.finetune.lora import apply_lora, merge_lora
from loom.model.pretrained import load_pretrained_gpt2
from loom.tokenizer import Tokenizer
from loom.train.checkpoint import save_checkpoint


def run(
    data_path: str,
    checkpoint_dir: str,
    num_epochs: int,
    batch_size: int,
    lr: float,
    max_length: int,
    rank: int,
    alpha: int,
):
    tok = Tokenizer()
    ds = InstructionDataset(data_path, tok)
    n = len(ds)
    train_ds = Subset(ds, range(int(0.85 * n)))
    val_ds = Subset(ds, range(int(0.85 * n), int(0.95 * n)))

    collate = functools.partial(instruction_collate_fn, max_length=max_length)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = GPT_CONFIG_124M_PRETRAINED
    model = load_pretrained_gpt2(cfg, "gpt2-small (124M)")
    model = apply_lora(model, LoRAConfig(rank=rank, alpha=alpha))
    model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable: {trainable:,} / {total:,}")

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    train_instruction_model(model, train_loader, val_loader, optimizer, device, num_epochs=num_epochs)

    model = merge_lora(model)
    path = save_checkpoint(model, optimizer, epoch=num_epochs - 1, global_step=-1, checkpoint_dir=checkpoint_dir)
    print(f"saved: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/raw/instruction/instruction-data.json")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/lora_instruct")
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    args = parser.parse_args()
    run(
        args.data,
        args.checkpoint_dir,
        args.num_epochs,
        args.batch_size,
        args.lr,
        args.max_length,
        args.rank,
        args.alpha,
    )
