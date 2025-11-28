import json
from io import BytesIO
from pathlib import Path
from typing import Optional

import requests
from PIL import Image
from torch.utils.data import Dataset


class ImageTextJsonlDataset(Dataset):
    def __init__(self, jsonl_path: str, processor, max_length: int = 40):
        self.jsonl_path = Path(jsonl_path)
        self.processor = processor
        self.max_length = max_length
        self.samples = []

        with self.jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                image = obj.get("image")
                text = obj.get("text")
                if image and text:
                    self.samples.append({"image": image, "text": text})

    def __len__(self):
        return len(self.samples)

    def _load_image(self, url: str) -> Optional[Image.Image]:
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            img = Image.open(BytesIO(resp.content)).convert("RGB")
            return img
        except Exception:
            return None

    def __getitem__(self, idx: int):
        # tránh đệ quy vô hạn khi nhiều ảnh liên tiếp bị lỗi
        attempts = 0
        max_attempts = 5
        num_samples = len(self.samples)

        while attempts < max_attempts:
            sample = self.samples[idx]
            img = self._load_image(sample["image"])
            if img is not None:
                text = sample["text"]

                inputs = self.processor(
                    images=img,
                    text=text,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                # remove batch dimension
                for k in inputs:
                    inputs[k] = inputs[k].squeeze(0)
                return inputs

            # nếu lỗi ảnh thì thử sample kế tiếp (không dùng đệ quy)
            attempts += 1
            idx = (idx + 1) % num_samples

        # nếu thử nhiều lần vẫn lỗi, raise để gọi code phía trên xử lý
        raise RuntimeError("Too many invalid image samples in a row")