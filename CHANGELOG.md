# Changelog

No git history predates this file (repo initialized after most of the work below was
already done) — entries reconstruct the actual build order from session notes so a new
reader can follow the same path the code was built in.

## Ch1-4 — Foundations

- Tokenizer wrapper (`loom/tokenizer`), sliding-window pretrain dataset/dataloader
  (`loom/dataset`).
- Attention built incrementally: `SelfAttention` -> `CausalAttention` ->
  `MultiHeadAttention` (`loom/model/attention.py`) — kept as separate classes rather
  than jumping straight to the final version, to make each mechanism inspectable.
- Transformer block + full GPT model (`loom/model/transformer_block.py`,
  `loom/model/gpt.py`). Forward pass verified end-to-end:
  `[batch, seq_len, vocab_size]`.

## Ch5 — Pretraining

- Pretraining loop with grad clipping + periodic sample generation
  (`loom/train/pretrain.py`).
- Checkpoint save/load/resume (`loom/train/checkpoint.py`).
- OpenAI GPT-2 pretrained weight loading (`loom/model/pretrained.py`) — maps HF's
  Conv1D-shaped state dict into this repo's `nn.Linear`-based architecture, handles
  QKV split + weight tying.

## Ch6 — Classification fine-tuning

- SMS spam/ham classifier (`loom/finetune/classifier.py`) — swaps `out_head` for a
  2-class head, freezes everything except the last transformer block + head.
- Result: 86.3% val accuracy after 1 epoch.

## Ch7 — Instruction fine-tuning + LoRA

- Alpaca-format instruction fine-tuning (`loom/finetune/instruction.py`) — full
  last-block fine-tune, 1100-example dataset.
- LoRA (`loom/finetune/lora.py`) — rank-8 adapters on attention q/v projections,
  24x fewer trainable params than the full last-block fine-tune (296K vs 7.09M).
- Eval utility (`scripts/run_eval.py`) — base vs fine-tuned checkpoint, same prompts,
  side by side.

## Hugging Face Hub publishing (this session)

- Built `hub_export/` — self-contained HF `PreTrainedModel`/`PretrainedConfig`
  wrapper (`configuration_loom.py`, `modeling_loom.py`) plus `export_to_hub.py`
  (parity-checks exported model against the original before saving/pushing).
- Discovered no meaningful "vanilla" checkpoint existed on disk
  (`checkpoints/smoketest/checkpoint.pt` was a from-scratch toy-data demo run, not
  real weights) — added `scripts/_save_vanilla_gpt2.py` to save real GPT-2 weights,
  zero fine-tuning, as a proper baseline checkpoint.
- Added `merge_lora()` to `loom/finetune/lora.py` — folds trained LoRA deltas back
  into plain `nn.Linear` layers so a LoRA-trained model's state_dict matches vanilla
  `GPTModel` exactly and can go through the same export pipeline.
- Added `scripts/_train_lora_instruct.py` — LoRA + instruction fine-tune combined
  (same Alpaca dataset as the full fine-tune, for direct comparison): 294,912
  trainable params, train loss 3.125 -> val loss 1.738, 1 epoch.
- Pushed three models, naming convention `loom-gpt2-124m[-variant]`:
  - `msclaw/loom-gpt2-124m` — vanilla, zero fine-tune
  - `msclaw/loom-gpt2-124m-instruct` — full fine-tune, Alpaca instructions
  - `msclaw/loom-gpt2-124m-lora-instruct` — LoRA, same task, merged before export
- Each repo ships its own model card, `configuration_loom.py`, `modeling_loom.py` so
  `trust_remote_code=True` works for anyone loading them independently.

## Not done yet

- `scripts/run_pretrain.py`, `run_finetune_classifier.py`,
  `run_finetune_instruction.py`, `run_lora.py` — empty CLI wrapper placeholders;
  underlying functions work and are exercised via inline scripts (see README).
- No `pyproject.toml` / editable install (`PYTHONPATH=.` required).
- No pytest suite — verification so far is inline scripts, shape/loss/accuracy checks
  logged in README's progress notes.
- LoRA + classification (binary spam/ham with adapters) — code path exists but
  `hub_export`'s wrapper is causal-LM-only; would need a second HF wrapper class
  (`AutoModelForSequenceClassification`) to publish.
