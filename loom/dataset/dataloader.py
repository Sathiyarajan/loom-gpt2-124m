from torch.utils.data import DataLoader

from loom.tokenizer import Tokenizer
from loom.dataset.pretrain_dataset import PretrainDataset


def create_pretrain_dataloader(
    text: str,
    context_length: int,
    batch_size: int = 4,
    stride: int | None = None,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    tokenizer = Tokenizer()
    stride = stride or context_length
    dataset = PretrainDataset(text, tokenizer, context_length, stride)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
