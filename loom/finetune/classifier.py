import pandas as pd
import torch
from torch.utils.data import Dataset

from loom.tokenizer import Tokenizer


class SpamDataset(Dataset):
    """Binary classification dataset: SMS text -> ham(0)/spam(1) label."""

    def __init__(self, csv_path: str, tokenizer: Tokenizer, max_length: int, pad_token_id: int = 50256):
        df = pd.read_csv(csv_path, sep="\t", header=None, names=["label", "text"])
        self.labels = (df["label"] == "spam").astype(int).tolist()
        encoded = [tokenizer.encode(text) for text in df["text"]]
        self.max_length = max_length or max(len(e) for e in encoded)
        self.encoded_texts = [
            e[: self.max_length] + [pad_token_id] * (self.max_length - len(e[: self.max_length]))
            for e in encoded
        ]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.encoded_texts[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


def add_classification_head(model, emb_dim: int, num_classes: int = 2):
    """Replace LM head with a classification head; freeze all but last block + norm + head."""
    for param in model.parameters():
        param.requires_grad = False

    model.out_head = torch.nn.Linear(emb_dim, num_classes)

    for param in model.trf_blocks[-1].parameters():
        param.requires_grad = True
    for param in model.final_norm.parameters():
        param.requires_grad = True
    for param in model.out_head.parameters():
        param.requires_grad = True

    return model


@torch.no_grad()
def calc_classification_accuracy(data_loader, model, device, num_batches: int | None = None):
    model.eval()
    correct, total = 0, 0
    num_batches = len(data_loader) if num_batches is None else min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        input_batch, target_batch = input_batch.to(device), target_batch.to(device)
        logits = model(input_batch)[:, -1, :]
        predicted = torch.argmax(logits, dim=-1)
        correct += (predicted == target_batch).sum().item()
        total += target_batch.numel()
    model.train()
    return correct / total


def calc_classification_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)[:, -1, :]
    return torch.nn.functional.cross_entropy(logits, target_batch)


def train_classifier(model, train_loader, val_loader, optimizer, device, num_epochs: int):
    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_classification_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()

        train_acc = calc_classification_accuracy(train_loader, model, device, num_batches=10)
        val_acc = calc_classification_accuracy(val_loader, model, device, num_batches=10)
        print(f"epoch {epoch+1}: train acc {train_acc:.3f}, val acc {val_acc:.3f}")

    return model
