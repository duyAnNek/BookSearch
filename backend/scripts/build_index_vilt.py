import json
from pathlib import Path

import faiss
import numpy as np

INDEX_DIR = Path("data/index_vilt")


def main():
    emb_path = INDEX_DIR / "embeddings.npy"
    meta_path = INDEX_DIR / "metadata.json"
    index_path = INDEX_DIR / "index_vilt.faiss"

    embs = np.load(emb_path)  # [N, D], đã normalize từ trước

    d = embs.shape[1]
    index = faiss.IndexFlatIP(d)  # dùng inner product với vector đã normalize
    index.add(embs)

    faiss.write_index(index, str(index_path))

    with meta_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    print("Built ViLT index:")
    print("  vectors:", embs.shape[0])
    print("  dim    :", d)
    print("  saved  :", index_path)


if __name__ == "__main__":
    main()