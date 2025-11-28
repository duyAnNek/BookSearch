import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import cfg
from .datasets import ImageTextJsonlDataset
from .models_vilt import load_vilt_retrieval


def train():
    os.makedirs(cfg.output_dir, exist_ok=True)

    processor, model, device = load_vilt_retrieval(cfg.model_name)

    train_ds = ImageTextJsonlDataset(cfg.train_file, processor)
    val_ds = ImageTextJsonlDataset(cfg.val_file, processor)

    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    for epoch in range(cfg.num_epochs):
        # ----- TRAIN -----
        pbar = tqdm(train_dl, desc=f"Epoch {epoch} [train]")
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}

            # tất cả image-text trong batch đều là cặp khớp (label = 1)
            batch_size = next(iter(batch.values())).size(0)
            labels = torch.ones(batch_size, dtype=torch.long, device=device)

            outputs = model(**batch, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            pbar.set_postfix({"loss": float(loss.item())})

        # ----- VALIDATION -----
        model.eval()
        val_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            for batch in tqdm(val_dl, desc=f"Epoch {epoch} [val]"):
                batch = {k: v.to(device) for k, v in batch.items()}
                batch_size = next(iter(batch.values())).size(0)
                labels = torch.ones(batch_size, dtype=torch.long, device=device)

                outputs = model(**batch, labels=labels)
                loss = outputs.loss
                val_loss += float(loss.item())
                val_steps += 1

        val_loss = val_loss / max(val_steps, 1)
        print(f"Epoch {epoch} validation loss: {val_loss:.4f}")
        model.train()

        torch.save(
            model.state_dict(),
            os.path.join(cfg.output_dir, f"epoch_{epoch}.pt"),
        )


if __name__ == "__main__":
    train()