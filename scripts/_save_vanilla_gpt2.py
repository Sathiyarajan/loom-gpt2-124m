import torch

from config import GPT_CONFIG_124M_PRETRAINED
from loom.model.pretrained import load_pretrained_gpt2
from loom.train.checkpoint import save_checkpoint

cfg = GPT_CONFIG_124M_PRETRAINED
model = load_pretrained_gpt2(cfg, "gpt2-small (124M)")
optimizer = torch.optim.AdamW(model.parameters())
path = save_checkpoint(model, optimizer, epoch=0, global_step=0, checkpoint_dir="checkpoints/vanilla_gpt2")
print("saved:", path)
