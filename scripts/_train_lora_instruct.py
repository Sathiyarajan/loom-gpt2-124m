import functools

import torch
from torch.utils.data import DataLoader, Subset

from config import GPT_CONFIG_124M_PRETRAINED, LoRAConfig
from loom.finetune.instruction import InstructionDataset, instruction_collate_fn, train_instruction_model
from loom.finetune.lora import apply_lora, merge_lora
from loom.model.pretrained import load_pretrained_gpt2
from loom.tokenizer import Tokenizer
from loom.train.checkpoint import save_checkpoint

device = "cuda" if torch.cuda.is_available() else "cpu"
cfg = GPT_CONFIG_124M_PRETRAINED

tok = Tokenizer()
ds = InstructionDataset("data/raw/instruction/instruction-data.json", tok)
n = len(ds)
train_ds = Subset(ds, range(int(0.85 * n)))
val_ds = Subset(ds, range(int(0.85 * n), int(0.95 * n)))

collate = functools.partial(instruction_collate_fn, max_length=256)
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate)
val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, collate_fn=collate)

model = load_pretrained_gpt2(cfg, "gpt2-small (124M)")
lora_cfg = LoRAConfig(rank=8, alpha=16)
model = apply_lora(model, lora_cfg)
model.to(device)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"trainable: {trainable:,} / {total:,}")

optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)
train_instruction_model(model, train_loader, val_loader, optimizer, device, num_epochs=1)

model = merge_lora(model)
path = save_checkpoint(model, optimizer, epoch=0, global_step=0, checkpoint_dir="checkpoints/lora_instruct")
print("saved:", path)
