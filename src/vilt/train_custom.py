import os
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import ViltProcessor, ViltModel

from ..config import cfg
from ..datasets import ImageTextJsonlDataset


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_model_and_processor():
    processor = ViltProcessor.from_pretrained(cfg.model_name)
    model = ViltModel.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.float32,  # luôn dùng float32 khi fine-tune
    )
    device = get_device()

    # KHÔNG load checkpoint cũ nữa
    # ckpt_dir = Path(cfg.output_dir)
    # if ckpt_dir.exists():
    #     ckpts = sorted(ckpt_dir.glob("vilt_custom_epoch_*.pt"))
    #     if ckpts:
    #         ckpt_path = ckpts[-1]
    #         state = torch.load(ckpt_path, map_location=device)
    #         model.load_state_dict(state)

    model.to(device)
    return processor, model, device


def contrastive_loss(similarity: torch.Tensor) -> torch.Tensor:
    batch_size = similarity.size(0)
    labels = torch.arange(batch_size, device=similarity.device)
    tau = 0.07
    logits = similarity / tau

    loss_i2t = torch.nn.functional.cross_entropy(logits, labels)
    loss_t2i = torch.nn.functional.cross_entropy(logits.t(), labels)
    return (loss_i2t + loss_t2i) / 2.0


def compute_similarity(embeds: torch.Tensor) -> torch.Tensor:
    normed = torch.nn.functional.normalize(embeds, dim=-1)
    return normed @ normed.t()


def train():
    os.makedirs(cfg.output_dir, exist_ok=True)

    print("Train config:", asdict(cfg))

    processor, model, device = build_model_and_processor()

    train_ds = ImageTextJsonlDataset(cfg.train_file, processor)
    val_ds = ImageTextJsonlDataset(cfg.val_file, processor)

    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    for epoch in range(cfg.num_epochs):
        model.train()
        pbar = tqdm(train_dl, desc=f"Epoch {epoch} [train]")
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            embeds = outputs.pooler_output  # [B, D]

            if torch.isnan(embeds).any() or torch.isinf(embeds).any():
                print("NaN/Inf in embeds at train batch")
                continue  # bỏ batch này

            sim = compute_similarity(embeds)

            if torch.isnan(sim).any() or torch.isinf(sim).any():
                print("NaN/Inf in similarity at train batch")
                continue

            loss = contrastive_loss(sim)

            if torch.isnan(loss) or torch.isinf(loss):
                print("NaN/Inf loss at train batch")
                continue

            sim = compute_similarity(embeds)
            loss = contrastive_loss(sim)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            pbar.set_postfix({"loss": float(loss.item())})

        model.eval()
        val_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            for batch in tqdm(val_dl, desc=f"Epoch {epoch} [val]"):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                embeds = outputs.pooler_output

                sim = compute_similarity(embeds)
                loss = contrastive_loss(sim)

                val_loss += float(loss.item())
                val_steps += 1

        val_loss = val_loss / max(val_steps, 1)
        print(f"Epoch {epoch} validation loss: {val_loss:.4f}")

        ckpt_path = os.path.join(cfg.output_dir, f"vilt_custom_epoch_{epoch}.pt")
        torch.save(model.state_dict(), ckpt_path)
        print("Saved checkpoint to", ckpt_path)


if __name__ == "__main__":
    train()
