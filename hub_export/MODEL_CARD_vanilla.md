---
license: mit
tags:
  - gpt2
  - from-scratch
  - educational
---

# loom-gpt2-124m

A from-scratch GPT-2-architecture implementation, initialized directly with OpenAI's
public GPT-2 small (124M) weights and **not fine-tuned at all**. Built as a personal
capstone project following Sebastian Raschka's *Build a Large Language Model (From
Scratch)* (Manning, 2024). Structural reference:
[rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch).

**This is an educational implementation, not a production model.** The architecture
(attention, transformer blocks, GPT model) was hand-written from the book rather than
imported from a library, then verified to correctly load and reproduce OpenAI's GPT-2
small checkpoint (parity-checked against the original weights, max logit diff < 1e-4).

This is the **vanilla baseline** in the `loom` model family — no fine-tuning of any
kind applied. See `msclaw/loom-gpt2-124m-instruct` (if/when pushed) for the
instruction-tuned variant built on top of this same base.

## Model description

- Architecture: decoder-only transformer, GPT-2-small shape (12 layers, 12 heads,
  768 embedding dim, 1024 context length, 124M parameters)
- Weights: OpenAI GPT-2 small, loaded via HF Hub (`gpt2`), unmodified
- Fine-tuning: none — this checkpoint is the base model only

## Hardware used

Weights loaded and exported on a single consumer laptop GPU: NVIDIA RTX 5060 Laptop,
8GB VRAM (Blackwell architecture, sm_120), inside WSL2 Ubuntu.

## Intended use

- Reference/teaching artifact: demonstrates that a from-scratch GPT implementation
  correctly loads and reproduces real GPT-2 weights.
- Baseline for comparison against fine-tuned variants in the same repo family.
- Not intended for production use, safety-critical applications, or as a general
  assistant.

## Limitations

- No RLHF/safety fine-tuning of any kind — outputs are raw GPT-2 base behavior only.
- `generate_simple()` is a minimal greedy/top-k/temperature sampler, not KV-cached —
  slower than standard HF `.generate()` and does not support beam search.
- Requires `trust_remote_code=True` to load (custom architecture, not a standard
  `transformers` model class).

## Usage

```python
from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained("msclaw/loom-gpt2-124m", trust_remote_code=True)
model.eval()

# tokenize with tiktoken (gpt2 BPE) -- same tokenizer the model was trained with
import tiktoken
enc = tiktoken.get_encoding("gpt2")
input_ids = torch.tensor([enc.encode("Every effort moves you")])

output_ids = model.generate_simple(input_ids, max_new_tokens=40)
print(enc.decode(output_ids[0].tolist()))
```

## Citation

If referencing this project, cite the underlying book:

> Raschka, S. (2024). *Build a Large Language Model (From Scratch)*. Manning.

Project source (full training pipeline, not just this exported checkpoint):
`https://github.com/msclaw/loom` *(update once repo is pushed to GitHub, if ever made
public — currently local-only)*.
