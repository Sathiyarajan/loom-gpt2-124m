---
license: mit
tags:
  - gpt2
  - from-scratch
  - educational
  - instruction-tuning
  - lora
---

# loom-gpt2-124m-lora-instruct

A from-scratch GPT-2-architecture language model, implemented as a personal capstone
project.

**This is an educational implementation, not a production model.** The architecture
(attention, transformer blocks, GPT model) was hand-written rather than imported from
a library, then initialized from OpenAI's public GPT-2 small (124M)
weights and instruction-fine-tuned via **LoRA** (Low-Rank Adaptation) on the same
Alpaca-format dataset used for `msclaw/loom-gpt2-124m-instruct`.

The trained LoRA adapters were merged back into the base weights before export, so
this repo ships a plain, standalone causal-LM checkpoint — no separate adapter-loading
step needed at inference time.

## Model description

- Architecture: decoder-only transformer, GPT-2-small shape (12 layers, 12 heads,
  768 embedding dim, 1024 context length, 124M parameters)
- Base weights: OpenAI GPT-2 small, loaded via HF Hub (`gpt2`)
- Fine-tuning: LoRA (rank 8, alpha 16) applied to attention q/k/v projections only,
  instruction-following, Alpaca prompt format (`### Instruction: ... ### Response: ...`)
- Trainable parameters during fine-tuning: 294,912 / 163,332,096 total
  (~0.18%, ~24x fewer than the full last-block fine-tune used for the
  `-instruct` variant)

## Training data

- 1100 examples, Alpaca-style instruction/input/output triples
- 85/10/5 train/val/test split
- 1 epoch, AdamW, lr=5e-5, batch_size=4, max sequence length 256 tokens (padded,
  loss masked on padding via `ignore_index=-100`)
- Result: train loss 3.125, val loss 1.738 after 1 epoch

## Hardware used

Trained on a single consumer laptop GPU: NVIDIA RTX 5060 Laptop, 8GB VRAM
(Blackwell architecture, sm_120), inside WSL2 Ubuntu.

## Intended use

- Reference/teaching artifact: demonstrates parameter-efficient fine-tuning (LoRA) on
  a from-scratch GPT implementation, and lets you compare quality/cost against the
  full-fine-tune `-instruct` sibling model.
- Not intended for production use, safety-critical applications, or as a general
  assistant — training data and fine-tuning volume are both small.

## Limitations

- Instruction fine-tuning used only 1 epoch on ~935 training examples — far less
  data/compute than production instruction-tuned models.
- LoRA applied to q/k/v projections only (not MLP layers) — matches the original LoRA
  paper's minimal-footprint configuration, not necessarily the highest-quality one.
- No RLHF/safety fine-tuning of any kind.
- `generate_simple()` is a minimal greedy/top-k/temperature sampler, not KV-cached —
  slower than standard HF `.generate()` and does not support beam search.
- Requires `trust_remote_code=True` to load (custom architecture, not a standard
  `transformers` model class).

## Usage

```python
from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained("msclaw/loom-gpt2-124m-lora-instruct", trust_remote_code=True)
model.eval()

import tiktoken
enc = tiktoken.get_encoding("gpt2")
input_ids = torch.tensor([enc.encode("### Instruction:\nName the capital of France.\n\n### Response:\n")])

output_ids = model.generate_simple(input_ids, max_new_tokens=40)
print(enc.decode(output_ids[0].tolist()))
```

## Related models

- `msclaw/loom-gpt2-124m` — vanilla base (no fine-tuning)
- `msclaw/loom-gpt2-124m-instruct` — same instruction data, full last-block fine-tune
  instead of LoRA

## Source

Project source (full training pipeline, not just this exported checkpoint):
`https://github.com/Sathiyarajan/loom-gpt2-124m`.
