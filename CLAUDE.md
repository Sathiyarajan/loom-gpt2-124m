# CLAUDE.md

Guidance for Claude Code sessions working in this repo.

## What this is

Personal capstone project: a GPT-2-style language model built from scratch (tokenizer,
attention, transformer blocks, pretraining, fine-tuning, LoRA), organized as a clean,
reusable reference implementation. Full details, architecture diagram, and per-stage
run instructions: see `README.md`.

## Environment (do not re-derive, already solved)

- Repo lives at `~/projects/loom` inside **WSL2 Ubuntu 26.04** — Linux filesystem only,
  never `/mnt/c`. Windows host only supplies the NVIDIA driver.
- GPU: RTX 5060 Laptop, 8GB VRAM, Blackwell/sm_120.
- Python 3.12.13 venv at `.venv`, installed via `uv` (not apt — deadsnakes PPA has no
  builds for Ubuntu 26.04's `resolute` codename).
- torch 2.11.0+cu128 — verified working CUDA (`torch.cuda.is_available()` True,
  capability `(12, 0)`).
- Always run with `PYTHONPATH=.` from repo root and `.venv` activated — package isn't
  installed editable yet (see TODOs).
- When editing files from a Windows-side Claude session, use the UNC path
  `\\wsl.localhost\Ubuntu\home\sathi\projects\loom\...` with Read/Edit/Write tools
  directly — reliable. Do NOT pipe multi-line file content through
  `wsl.exe -d Ubuntu -- bash -c "... << 'EOF' ..."` heredocs from an outer shell —
  backticks/`$()` inside get expanded by the OUTER shell before reaching the inner one,
  silently corrupting file content. This bit us once already (see README's
  "Issues encountered" section).

## Current state (all verified working, this session)

- **Ch1-4**: tokenizer (`loom/tokenizer`), sliding-window dataset/dataloader
  (`loom/dataset`), attention built incrementally — Self -> Causal -> MultiHead
  (`loom/model/attention.py`), transformer block + full GPT model
  (`loom/model/transformer_block.py`, `loom/model/gpt.py`). Forward pass verified
  `[batch, seq_len, vocab_size]`.
- **Ch5**: pretraining loop with grad clipping + periodic sample generation
  (`loom/train/pretrain.py`), checkpoint save/load/resume (`loom/train/checkpoint.py`),
  pretrained OpenAI GPT-2 weight loading (`loom/model/pretrained.py` — maps HF's
  Conv1D-shaped state dict into our nn.Linear-based architecture, handles QKV split +
  weight tying).
- **Ch6**: classification fine-tuning (`loom/finetune/classifier.py`) — SMS spam/ham,
  86.3% val accuracy after 1 epoch, only last transformer block + head trainable.
- **Ch7**: instruction fine-tuning (`loom/finetune/instruction.py`, Alpaca format) +
  LoRA (`loom/finetune/lora.py`, q/v-projection adapters, 24x fewer trainable params
  than full last-block fine-tune: 296K vs 7.09M).
- **Eval**: `scripts/run_eval.py` — base vs fine-tuned checkpoint, same prompts,
  side by side.

## Config (single source of truth: config.py)

Two families of presets:
- `GPT_CONFIG_124M/355M/774M/1558M` — `context_length=256` (base 124M) or 1024 (rest),
  for from-scratch pretraining sized to fit 8GB VRAM.
- `GPT_CONFIG_124M/355M_PRETRAINED` — `context_length=1024`, `qkv_bias=True`, required
  shape to load real GPT-2 weights (position-embedding table is fixed at 1024, can't
  shrink without retraining).

`ACTIVE_MODEL_CONFIG` / `ACTIVE_TRAIN_CONFIG` / `ACTIVE_LORA_CONFIG` at the bottom of
`config.py` are what most code should default to reading — change these three lines to
rescale everything.

## What's NOT done yet (don't assume these exist)

- `scripts/run_pretrain.py`, `run_finetune_classifier.py`, `run_finetune_instruction.py`,
  `run_lora.py` are empty placeholder files — CLI argparse wrappers are still TODO. The
  underlying functions they'd call are implemented and tested (see README's numbered
  onboarding walkthrough for the equivalent inline invocations).
- No `pyproject.toml` / editable install — `PYTHONPATH=.` is a manual requirement.
- No unit test suite — verification so far is inline scripts per module, logged in
  README's Progress section, not preserved as pytest.
- No mixed precision / gradient accumulation — training loop is plain fp32.
- No weight tying in the from-scratch pretraining path (only the pretrained-loading
  path ties `tok_emb`/`out_head`, matching real GPT-2).

## Conventions established so far

- One class/function per concern, incremental build style (e.g. attention built as 3
  separate classes, not jumped straight to MultiHeadAttention) — keep this pattern for
  any new model components.
- Inline "why" comments only where a design choice isn't obvious from reading the code
  (e.g. why LayerNorm/GELU are custom instead of `torch.nn` builtins — matches GPT-2
  checkpoint param names/exact activation for pretrained-weight loading). Don't add
  "what it does" comments — names should carry that.
- Every new module gets a quick inline verification snippet run before being considered
  done (shape checks, loss going down, accuracy above baseline) — this repo has no
  pytest suite, so this is the only test coverage that exists; keep doing it for new
  code until a real test suite is added.
