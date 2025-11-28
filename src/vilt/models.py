import torch
from transformers import ViltProcessor, ViltForImageAndTextRetrieval


def load_vilt_retrieval(model_name: str = "dandelin/vilt-b32-mlm"):
    processor = ViltProcessor.from_pretrained(model_name)
    model = ViltForImageAndTextRetrieval.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return processor, model, device
