import json
from pathlib import Path
from typing import List, Dict

import faiss
import numpy as np
import torch
from PIL import Image
from transformers import ViltProcessor, ViltModel


class ViltSearchService:
    """Search service using fine-tuned ViLT embeddings and FAISS index.

    This service is independent from the existing CLIP-based SearchService
    and uses metadata built from train_image_text.jsonl instead of the DB.
    """

    def __init__(self, index_dir: str, model_name: str = "dandelin/vilt-b32-mlm") -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.index_dir = Path(index_dir)
        self.index_path = self.index_dir / "index_vilt.faiss"
        self.meta_path = self.index_dir / "metadata.json"

        self.model_name = model_name

        self.index: faiss.Index | None = None
        self.metadata: List[Dict] = []
        self.processor: ViltProcessor | None = None
        self.model: ViltModel | None = None

        # simple dummy image used to pair with text queries
        self.dummy_image = Image.new("RGB", (224, 224), color=(255, 255, 255))

        self._load_resources()

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _load_resources(self) -> None:
        if not self.index_path.exists():
            raise FileNotFoundError(f"ViLT FAISS index not found: {self.index_path}")
        if not self.meta_path.exists():
            raise FileNotFoundError(f"ViLT metadata not found: {self.meta_path}")

        # load metadata
        with self.meta_path.open("r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        # load FAISS index
        self.index = faiss.read_index(str(self.index_path))

        # load ViLT model + processor
        self.processor = ViltProcessor.from_pretrained(self.model_name)
        self.model = ViltModel.from_pretrained(self.model_name)

        # try to load latest fine-tuned checkpoint (same as used for index build)
        ckpt_root = Path("/data_root/outputs/checkpoints/vilt_vi")
        if ckpt_root.exists():
            ckpts = sorted(ckpt_root.glob("vilt_custom_epoch_*.pt"))
            if ckpts:
                ckpt_path = ckpts[-1]
                state = torch.load(ckpt_path, map_location=self.device)
                self.model.load_state_dict(state)

        self.model.to(self.device)
        self.model.eval()

    def _embed_text(self, text: str) -> np.ndarray:
        if self.processor is None or self.model is None:
            raise RuntimeError("ViLT model or processor not initialized")

        with torch.no_grad():
            inputs = self.processor(
                images=self.dummy_image,
                text=text,
                padding="max_length",
                truncation=True,
                max_length=40,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            outputs = self.model(**inputs)
            emb = outputs.pooler_output  # [1, D]
            emb = torch.nn.functional.normalize(emb, dim=-1)

        return emb.cpu().numpy()  # shape [1, D]

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def search_by_text(self, text_query: str, top_k: int = 20) -> List[Dict]:
        if self.index is None:
            raise RuntimeError("FAISS index not initialized")

        query_vec = self._embed_text(text_query)  # [1, D]
        distances, indices = self.index.search(query_vec, top_k)

        results: List[Dict] = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:
                continue
            if idx < 0 or idx >= len(self.metadata):
                continue

            meta = self.metadata[idx]
            results.append(
                {
                    "index": int(idx),
                    "score": float(distances[0][i]),
                    "image": meta.get("image"),
                    "text": meta.get("text"),
                }
            )

        return results
