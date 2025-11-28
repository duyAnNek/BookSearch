import json
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import cfg
from ..datasets import ImageTextJsonlDataset
from .train_custom import build_model_and_processor


def compute_embeddings(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(cfg.train_file)
    if not data_path.exists():
        raise FileNotFoundError(f"Train file not found: {data_path}")

    processor, model, device = build_model_and_processor()

    dataset = ImageTextJsonlDataset(str(data_path), processor)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False)

    all_embs: List[np.ndarray] = []
    metadata: List[Dict] = []

    model.eval()
    with torch.no_grad():
        idx_start = 0
        for batch in tqdm(dataloader, desc="Computing embeddings"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            emb = outputs.pooler_output
            emb = torch.nn.functional.normalize(emb, dim=-1)

            all_embs.append(emb.cpu().numpy())

            batch_size = emb.size(0)
            for i in range(batch_size):
                sample_idx = idx_start + i
                if sample_idx < len(dataset.samples):
                    sample = dataset.samples[sample_idx]
                    metadata.append(
                        {
                            "index": sample_idx,
                            "image": sample.get("image"),
                            "text": sample.get("text"),
                        }
                    )
                else:
                    metadata.append({"index": sample_idx})
            idx_start += batch_size

    embs_arr = np.concatenate(all_embs, axis=0)

    emb_path = output_dir / "embeddings.npy"
    meta_path = output_dir / "metadata.json"

    np.save(emb_path, embs_arr)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Saved embeddings to {emb_path}")
    print(f"Saved metadata to {meta_path}")
    print("Total vectors:", embs_arr.shape[0])


def main():
    out_dir = Path("data/index_vilt")
    compute_embeddings(out_dir)


if __name__ == "__main__":
    main()
