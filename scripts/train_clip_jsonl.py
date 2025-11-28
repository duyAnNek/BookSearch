import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from transformers import CLIPModel, CLIPProcessor

# Thêm thư mục gốc project vào sys.path để import src.* khi chạy từ scripts/
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.datasets import ImageTextJsonlDataset
DATA_PATH = BASE_DIR / "data_extend" / "train_image_text_all.jsonl"
OUTPUT_DIR = BASE_DIR / "model_clip"

BATCH_SIZE = 16
NUM_EPOCHS = 10
LR = 5e-6
VAL_RATIO = 0.05  # 5% cho validation
MODEL_NAME = "openai/clip-vit-base-patch32"


def create_dataloaders():
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    dataset = ImageTextJsonlDataset(str(DATA_PATH), processor)

    if len(dataset) == 0:
        raise RuntimeError(f"Dataset rỗng: {DATA_PATH}")

    val_size = max(1, int(len(dataset) * VAL_RATIO))
    train_size = len(dataset) - val_size

    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    return processor, train_dl, val_dl


def evaluate_retrieval(model: CLIPModel, processor: CLIPProcessor, val_dl, device: torch.device):
    model.eval()
    sim_scores = []
    hit1, hit5, total = 0, 0, 0

    with torch.no_grad():
        for batch in tqdm(val_dl, desc="[eval]"):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"],
                output_hidden_states=False,
            )

            image_embeds = outputs.image_embeds  # (B, D)
            text_embeds = outputs.text_embeds    # (B, D)

            image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

            # cosine similarity matrix (B, B)
            sim = text_embeds @ image_embeds.t()

            # đánh giá retrieval text->image trong batch
            # ground truth: ảnh đúng nằm cùng chỉ số
            _, indices = sim.topk(k=min(5, sim.size(1)), dim=-1)  # (B, k)

            bsz = sim.size(0)
            total += bsz
            for i in range(bsz):
                topk = indices[i].tolist()
                if i < len(topk) and topk[0] == i:
                    hit1 += 1
                if i in topk:
                    hit5 += 1

            sim_scores.append(sim.detach().cpu())

    hit1_acc = hit1 / max(total, 1)
    hit5_acc = hit5 / max(total, 1)

    return {
        "hit@1": hit1_acc,
        "hit@5": hit5_acc,
    }


def train():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    processor, train_dl, val_dl = create_dataloaders()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = CLIPModel.from_pretrained(MODEL_NAME)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    for epoch in range(1, NUM_EPOCHS + 1):
        # ----- TRAIN -----
        model.train()
        running_loss = 0.0
        steps = 0

        pbar = tqdm(train_dl, desc=f"Epoch {epoch} [train]")
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"],
                return_loss=True,
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            steps += 1
            pbar.set_postfix({"loss": running_loss / steps})

        train_loss = running_loss / max(steps, 1)

        # ----- EVAL -----
        metrics = evaluate_retrieval(model, processor, val_dl, device)

        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f}, "
            f"hit@1={metrics['hit@1']:.4f}, hit@5={metrics['hit@5']:.4f}"
        )

        # Lưu model mỗi epoch
        epoch_dir = OUTPUT_DIR / f"epoch_{epoch}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(epoch_dir)
        processor.save_pretrained(epoch_dir)

    # Lưu model cuối cùng
    final_dir = OUTPUT_DIR / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    print("✅ Đã lưu model CLIP cuối cùng tại", final_dir)


if __name__ == "__main__":
    train()
