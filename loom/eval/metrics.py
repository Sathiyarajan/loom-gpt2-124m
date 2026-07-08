import torch

from loom.tokenizer import Tokenizer


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches: int | None = None):
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    num_batches = len(data_loader) if num_batches is None else min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()
    return total_loss / num_batches


@torch.no_grad()
def generate_text(model, idx, max_new_tokens: int, context_length: int, temperature: float = 0.0, top_k: int | None = None):
    """Autoregressive generation. temperature=0.0 -> greedy decode."""
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_length:]
        logits = model(idx_cond)
        logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(-torch.inf).to(logits.device), logits)

        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        idx = torch.cat((idx, idx_next), dim=1)
    return idx


@torch.no_grad()
def generate_and_print_sample(model, device, start_context: str, context_length: int):
    tokenizer = Tokenizer()
    model.eval()
    encoded = torch.tensor(tokenizer.encode(start_context)).unsqueeze(0).to(device)
    out_ids = generate_text(model, encoded, max_new_tokens=30, context_length=context_length)
    decoded = tokenizer.decode(out_ids.squeeze(0).tolist())
    model.train()
    print(decoded.replace("\n", " "))
