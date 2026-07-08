---
license: mit
tags:
  - gpt2
  - from-scratch
  - educational
  - instruction-tuning
---

# loom-gpt2-124m-instruct

A from-scratch GPT-2-architecture language model, implemented and fine-tuned as a
personal capstone project following Sebastian Raschka's *Build a Large Language Model
(From Scratch)* (Manning, 2024). Structural reference:
[rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch).

**This is an educational implementation, not a production model.** The architecture
(attention, transformer blocks, GPT model) was hand-written from the book rather than
imported from a library, then initialized from OpenAI's public GPT-2 small (124M)
weights and instruction-fine-tuned on a small (1100-example) Alpaca-format dataset.

## Model description

- Architecture: decoder-only transformer, GPT-2-small shape (12 layers, 12 heads,
  768 embedding dim, 1024 context length, 124M parameters)
- Base weights: OpenAI GPT-2 small, loaded via HF Hub (`gpt2`)
- Fine-tuning: instruction-following, Alpaca prompt format
  (`### Instruction: ... ### Response: ...`)

## Training data

- **Instruction fine-tuning**: 1100 examples, Alpaca-style instruction/input/output
  triples (source: rasbt/LLMs-from-scratch ch07 dataset)
- 85/10/5 train/val/test split
- 1 epoch, AdamW, lr=5e-5, batch_size=4, max sequence length 256 tokens (padded,
  loss masked on padding via `ignore_index=-100`)

## Hardware used

Trained on a single consumer laptop GPU: NVIDIA RTX 5060 Laptop, 8GB VRAM
(Blackwell architecture, sm_120), inside WSL2 Ubuntu. Not trained on a cluster or
multiple GPUs — this reflects a hobbyist/educational compute budget, not
research-scale training.

## Intended use

- Reference/teaching artifact: demonstrates that a from-scratch GPT implementation
  can correctly load real GPT-2 weights and be fine-tuned to follow instructions.
- Not intended for production use, safety-critical applications, or as a general
  assistant — training data and fine-tuning volume are both small.

## Limitations

- Instruction fine-tuning used only 1 epoch on ~935 training examples — far less
  data/compute than production instruction-tuned models.
- No RLHF/safety fine-tuning of any kind — outputs are unfiltered base+instruction-SFT
  behavior only.
- `generate_simple()` is a minimal greedy/top-k/temperature sampler, not KV-cached —
  slower than standard HF `.generate()` and does not support beam search.
- Requires `trust_remote_code=True` to load (custom architecture, not a standard
  `transformers` model class).

## Usage

```python
from transformers import AutoConfig, AutoModelForCausalLM
from transformers import GPT2Tokenizer  # or tiktoken directly, see below

model = AutoModelForCausalLM.from_pretrained("msclaw/loom-gpt2-124m-instruct", trust_remote_code=True)
model.eval()

# tokenize with tiktoken (gpt2 BPE) -- same tokenizer the model was trained with
import tiktoken
enc = tiktoken.get_encoding("gpt2")
input_ids = torch.tensor([enc.encode("### Instruction:\nName the capital of France.\n\n### Response:\n")])

output_ids = model.generate_simple(input_ids, max_new_tokens=40)
print(enc.decode(output_ids[0].tolist()))
```

## Citation

If referencing this project, cite the underlying book:

> Raschka, S. (2024). *Build a Large Language Model (From Scratch)*. Manning.

Project source (full training pipeline, not just this exported checkpoint):
`https://github.com/YOUR_USERNAME/loom` *(update once repo is pushed to GitHub, if
ever made public — currently local-only)*.
