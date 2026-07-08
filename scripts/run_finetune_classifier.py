"""Spam/ham classification fine-tuning CLI. See README onboarding step 8 for the
equivalent inline invocation this wraps."""
import argparse
import tempfile

import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import GPT_CONFIG_124M_PRETRAINED
from loom.finetune.classifier import SpamDataset, add_classification_head, train_classifier
from loom.model.pretrained import load_pretrained_gpt2
from loom.tokenizer import Tokenizer
from loom.train.checkpoint import save_checkpoint


def run(data_path: str, checkpoint_dir: str, num_epochs: int, batch_size: int, lr: float, max_length: int, seed: int):
    df = pd.read_csv(data_path, sep="\t", header=None, names=["label", "text"])
    spam = df[df.label == "spam"]
    ham = df[df.label == "ham"].sample(len(spam), random_state=seed)
    balanced = pd.concat([spam, ham]).sample(frac=1, random_state=seed).reset_index(drop=True)
    n = len(balanced)

    with tempfile.NamedTemporaryFile(suffix=".tsv") as train_f, tempfile.NamedTemporaryFile(suffix=".tsv") as val_f:
        balanced[: int(0.8 * n)].to_csv(train_f.name, sep="\t", header=False, index=False)
        balanced[int(0.8 * n) :].to_csv(val_f.name, sep="\t", header=False, index=False)

        tok = Tokenizer()
        train_loader = DataLoader(SpamDataset(train_f.name, tok, max_length=max_length), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(SpamDataset(val_f.name, tok, max_length=max_length), batch_size=batch_size, shuffle=False)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg = GPT_CONFIG_124M_PRETRAINED
        model = load_pretrained_gpt2(cfg, "gpt2-small (124M)")
        model = add_classification_head(model, emb_dim=cfg.emb_dim, num_classes=2)
        model.to(device)

        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        train_classifier(model, train_loader, val_loader, optimizer, device, num_epochs=num_epochs)
        path = save_checkpoint(model, optimizer, epoch=num_epochs - 1, global_step=-1, checkpoint_dir=checkpoint_dir)
        print(f"saved: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/raw/sms_spam/SMSSpamCollection")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/classifier")
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    run(args.data, args.checkpoint_dir, args.num_epochs, args.batch_size, args.lr, args.max_length, args.seed)
