"""From-scratch pretraining CLI. See README onboarding step 6 for the equivalent
inline invocation this wraps."""
import argparse

import torch

from config import GPT_CONFIG_124M, TrainConfig
from loom.dataset.dataloader import create_pretrain_dataloader
from loom.model.gpt import GPTModel
from loom.train.pretrain import train_model
from loom.utils.seed import set_seed


def run(
    corpus_path: str,
    checkpoint_dir: str,
    num_epochs: int,
    eval_freq: int,
    eval_iter: int,
    batch_size: int,
    lr: float,
    val_split: float,
    seed: int,
):
    set_seed(seed)
    text = open(corpus_path).read()
    split = int(val_split * len(text))
    cfg = GPT_CONFIG_124M

    train_loader = create_pretrain_dataloader(text[:split], context_length=cfg.context_length, batch_size=batch_size)
    val_loader = create_pretrain_dataloader(text[split:], context_length=cfg.context_length, batch_size=batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPTModel(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    train_cfg = TrainConfig(
        num_epochs=num_epochs, eval_freq=eval_freq, eval_iter=eval_iter, checkpoint_dir=checkpoint_dir
    )
    train_model(model, train_loader, val_loader, optimizer, device, train_cfg, cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, default="data/raw/the-verdict.txt")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/pretrain")
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--eval-freq", type=int, default=5)
    parser.add_argument("--eval-iter", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--val-split", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    run(
        args.corpus,
        args.checkpoint_dir,
        args.num_epochs,
        args.eval_freq,
        args.eval_iter,
        args.batch_size,
        args.lr,
        args.val_split,
        args.seed,
    )
