import sys
import csv
from io import BytesIO
from pathlib import Path
from typing import List, Tuple

import requests
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "fine_tuned_clip_v2" / "epoch_9"
DATA_PATH = BASE_DIR / "Book.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_ITEMS = 500  # số mẫu ảnh-text để build gallery test
TOP_K = 5


def load_samples(max_items: int) -> List[dict]:
    """Đọc một phần dữ liệu để test.

    Hỗ trợ:
    - JSONL: các field `image`, `text` (dataset mới).
    - CSV: file Book.csv với cột `image_url`, `title`, `description`.
    """
    samples: List[dict] = []
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {DATA_PATH}")

    suffix = DATA_PATH.suffix.lower()

    # Trường hợp JSONL (dataset train_image_text_all.jsonl)
    if suffix == ".jsonl":
        with DATA_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    import json

                    obj = json.loads(line)
                except Exception:
                    continue
                img = obj.get("image")
                text = obj.get("text")
                if not img or not text:
                    continue
                samples.append({"image": img, "text": text})
                if len(samples) >= max_items:
                    break

    # Trường hợp CSV (Book.csv)
    elif suffix == ".csv":
        with DATA_PATH.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img = (row.get("image_url") or "").strip()
                title = (row.get("title") or "").strip()
                desc = (row.get("description") or "").strip()
                if not img or not title:
                    continue
                text = title
                if desc:
                    text = f"{title}. {desc}"
                samples.append({"image": img, "text": text})
                if len(samples) >= max_items:
                    break

    return samples


def download_image(url: str) -> Image.Image | None:
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        return img
    except Exception:
        return None


def build_image_gallery(processor: CLIPProcessor, samples: List[dict]) -> Tuple[torch.Tensor, List[str]]:
    """Encode ảnh thành embedding, trả về tensor (N, D) và list text tương ứng."""
    images: List[Image.Image] = []
    texts: List[str] = []

    for s in tqdm(samples, desc="Tải ảnh gallery"):
        img = download_image(s["image"])
        if img is None:
            continue
        images.append(img)
        texts.append(s["text"])

    if not images:
        raise RuntimeError("Không tải được ảnh nào để test.")

    all_embeds: List[torch.Tensor] = []
    batch_size = 16
    for i in tqdm(range(0, len(images), batch_size), desc="Encode ảnh"):
        batch_imgs = images[i : i + batch_size]
        inputs = processor(images=batch_imgs, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model.get_image_features(**inputs)  # (B, D) trên DEVICE
        all_embeds.append(outputs)

    image_embeds = torch.cat(all_embeds, dim=0)  # (N, D) trên DEVICE
    image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)

    return image_embeds, texts


def pretty_title(text: str, max_len: int = 80) -> str:
    t = text.split(".")[0].strip()
    if len(t) > max_len:
        t = t[: max_len - 3] + "..."
    return t


if __name__ == "main__":
    print("Script này nên được chạy như module: python scripts/test_clip_query.py")


def main() -> None:
    global model

    print("Đang load model từ", MODEL_DIR)
    model = CLIPModel.from_pretrained(str(MODEL_DIR)).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(str(MODEL_DIR))

    print("Đang đọc dữ liệu từ", DATA_PATH)
    samples = load_samples(MAX_ITEMS)
    print(f"Đã lấy {len(samples)} mẫu để build gallery.")

    image_embeds, texts = build_image_gallery(processor, samples)

    while True:
        try:
            query = input("\nNhập câu truy vấn (Enter để thoát): ").strip()
        except EOFError:
            break
        if not query:
            break

        inputs = processor(text=[query], return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            text_embeds = model.get_text_features(**inputs)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        # similarity với tất cả ảnh
        sim = (text_embeds @ image_embeds.t()).squeeze(0)  # (N,)
        topk = min(TOP_K, sim.size(0))
        values, indices = torch.topk(sim, k=topk)

        print(f"Top {topk} gợi ý:")
        for rank, (idx, score) in enumerate(zip(indices.tolist(), values.tolist()), start=1):
            print(f"  {rank}. score={score:.3f} | {pretty_title(texts[idx])}")


if __name__ == "__main__":
    main()
