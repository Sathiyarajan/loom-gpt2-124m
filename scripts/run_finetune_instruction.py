"""Alpaca-format instruction fine-tuning CLI. See README onboarding step 9 for the
equivalent inline invocation this wraps."""
import argparse
import functools

import torch
from torch.utils.data import DataLoader, Subset

from config import GPT_CONFIG_124M_PRETRAINED
from loom.finetune.instruction import InstructionDataset, instruction_collate_fn, train_instruction_model
from loom.model.pretrained import load_pretrained_gpt2
from loom.tokenizer import Tokenizer
from loom.train.checkpoint import save_checkpoint


def run(data_path: str, checkpoint_dir: str, num_epochs: int, batch_size: int, lr: float, max_length: int):
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
    model = load_pretrained_gpt2(cfg, "gpt2-small (124M)").to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    train_instruction_model(model, train_loader, val_loader, optimizer, device, num_epochs=num_epochs)
    path = save_checkpoint(model, optimizer, epoch=num_epochs - 1, global_step=-1, checkpoint_dir=checkpoint_dir)
    print(f"saved: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/raw/instruction/instruction-data.json")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/instruction_ft")
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()
    run(args.data, args.checkpoint_dir, args.num_epochs, args.batch_size, args.lr, args.max_length)
