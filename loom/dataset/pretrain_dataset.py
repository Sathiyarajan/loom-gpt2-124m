import torch
from torch.utils.data import Dataset

from loom.tokenizer import Tokenizer


class PretrainDataset(Dataset):
    """Sliding-window next-token dataset over a raw text corpus."""

    def __init__(self, text: str, tokenizer: Tokenizer, context_length: int, stride: int):
        token_ids = tokenizer.encode(text)
        self.input_ids = []
        self.target_ids = []
        for i in range(0, len(token_ids) - context_length, stride):
            self.input_ids.append(torch.tensor(token_ids[i : i + context_length]))
            self.target_ids.append(torch.tensor(token_ids[i + 1 : i + context_length + 1]))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
