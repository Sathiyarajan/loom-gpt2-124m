import torch

from config import GPTConfig, TrainConfig
from loom.eval.metrics import calc_loss_batch, calc_loss_loader, generate_and_print_sample
from loom.train.checkpoint import save_checkpoint


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    train_cfg: TrainConfig,
    model_cfg: GPTConfig,
    start_context: str = "Every effort moves you",
):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(train_cfg.num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % train_cfg.eval_freq == 0:
                model.eval()
                with torch.no_grad():
                    train_loss = calc_loss_loader(train_loader, model, device, num_batches=train_cfg.eval_iter)
                    val_loss = calc_loss_loader(val_loader, model, device, num_batches=train_cfg.eval_iter)
                model.train()
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"epoch {epoch+1} step {global_step:06d}: train loss {train_loss:.3f}, val loss {val_loss:.3f}")

        generate_and_print_sample(model, device, start_context, model_cfg.context_length)
        save_checkpoint(model, optimizer, epoch, global_step, train_cfg.checkpoint_dir)

    return train_losses, val_losses, track_tokens_seen
