from pathlib import Path
from io import BytesIO

import torch
import torch.nn.functional as F
import requests
from PIL import Image
from transformers import ViltProcessor, ViltModel

from ..config import cfg


def load_model(ckpt_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = ViltProcessor.from_pretrained(cfg.model_name)
    model = ViltModel.from_pretrained(cfg.model_name)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return processor, model, device


def load_image_from_url(url: str) -> Image.Image:
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return Image.open(BytesIO(resp.content)).convert("RGB")


def embed_image_text(processor, model, device, image_url: str, text: str) -> torch.Tensor:
    img = load_image_from_url(image_url)
    inputs = processor(
        images=img,
        text=text,
        padding="max_length",
        truncation=True,
        max_length=40,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        emb = outputs.pooler_output
        emb = F.normalize(emb, dim=-1)
    return emb.squeeze(0)


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=-1).item())


def get_latest_checkpoint() -> Path:
    out_dir = Path(cfg.output_dir)
    if not out_dir.exists():
        raise FileNotFoundError(f"Output dir not found: {out_dir}")

    ckpts = sorted(out_dir.glob("vilt_custom_epoch_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {out_dir}")
    return ckpts[-1]


def main():
    ckpt_path = get_latest_checkpoint()
    print("Loading checkpoint:", ckpt_path)

    processor, model, device = load_model(str(ckpt_path))
    print("Device:", device)

    image_url = "https://salt.tikicdn.com/cache/280x280/ts/product/6b/dc/23/c5ea3090c33c8dfe9f08c0ecd44290dd.jpg"
    text1 = "Tô màu phát triển não bộ cho bé 1-5 tuổi"
    text2 = "Cuốn sách về đầu tư chứng khoán cho người lớn"

    emb1 = embed_image_text(processor, model, device, image_url, text1)
    emb2 = embed_image_text(processor, model, device, image_url, text2)

    sim_11 = cosine_sim(emb1, emb1)
    sim_12 = cosine_sim(emb1, emb2)

    print("Similarity(image, text1 vs itself):", sim_11)
    print("Similarity(image, text1 vs text2):", sim_12)

    if sim_11 > sim_12:
        print("=> text1 phù hợp với ảnh hơn text2 (kết quả tốt).")
    else:
        print("=> text2 lại gần hơn text1 – cần xem lại chất lượng model/dữ liệu.")


if __name__ == "__main__":
    main()
