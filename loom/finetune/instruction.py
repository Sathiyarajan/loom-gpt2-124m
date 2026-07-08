import json

import torch
from torch.utils.data import Dataset

from loom.tokenizer import Tokenizer

ALPACA_PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}"
)

ALPACA_INPUT_SUFFIX = "\n\n### Input:\n{input}"
ALPACA_RESPONSE_SUFFIX = "\n\n### Response:\n"


def format_prompt(example: dict) -> str:
    prompt = ALPACA_PROMPT_TEMPLATE.format(instruction=example["instruction"])
    if example.get("input"):
        prompt += ALPACA_INPUT_SUFFIX.format(input=example["input"])
    return prompt + ALPACA_RESPONSE_SUFFIX


class InstructionDataset(Dataset):
    """Alpaca-style instruction/input/output examples, tokenized as prompt+response."""

    def __init__(self, json_path: str, tokenizer: Tokenizer):
        with open(json_path) as f:
            self.data = json.load(f)
        self.encoded_texts = []
        for example in self.data:
            full_text = format_prompt(example) + example["output"]
            self.encoded_texts.append(tokenizer.encode(full_text))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.encoded_texts[idx]


def instruction_collate_fn(batch, pad_token_id: int = 50256, ignore_index: int = -100, max_length: int | None = None):
    """Pad batch to same length; targets = inputs shifted by 1, padding masked with ignore_index."""
    batch_max_length = max(len(item) + 1 for item in batch)
    if max_length is not None:
        batch_max_length = min(batch_max_length, max_length)

    inputs_list, targets_list = [], []
    for item in batch:
        padded = item + [pad_token_id] * (batch_max_length - len(item))
        input_ids = torch.tensor(padded[:-1])
        target_ids = torch.tensor(padded[1:])
        mask = target_ids == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            target_ids[indices[1:]] = ignore_index
        inputs_list.append(input_ids)
        targets_list.append(target_ids)

    return torch.stack(inputs_list), torch.stack(targets_list)


def calc_instruction_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten(), ignore_index=-100
    )
    return loss


def train_instruction_model(model, train_loader, val_loader, optimizer, device, num_epochs: int):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_instruction_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for input_batch, target_batch in val_loader:
                val_loss += calc_instruction_loss_batch(input_batch, target_batch, model, device).item()
        model.train()

        print(f"epoch {epoch+1}: train loss {total_loss/len(train_loader):.3f}, val loss {val_loss/len(val_loader):.3f}")

    return model
