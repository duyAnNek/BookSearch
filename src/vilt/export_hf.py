from pathlib import Path

import torch
from transformers import ViltProcessor, ViltModel

from ..config import cfg


def get_latest_checkpoint() -> Path:
    out_dir = Path(cfg.output_dir)
    if not out_dir.exists():
        raise FileNotFoundError(f"Output dir not found: {out_dir}")

    ckpts = sorted(out_dir.glob("vilt_custom_epoch_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {out_dir}")
    return ckpts[-1]


def export_to_hf_folder(output_folder: str = "models/vilt_vi_finetuned") -> None:
    """Load latest fine-tuned ViLT checkpoint and save in HF format.

    This will create a folder like the CLIP model folder, with:
      - config.json
      - preprocessor_config.json
      - special_tokens_map.json
      - tokenizer.json / tokenizer_config.json / vocab.json
      - model.safetensors
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = get_latest_checkpoint()
    print("Using checkpoint:", ckpt_path)

    # load base model + processor
    processor = ViltProcessor.from_pretrained(cfg.model_name)
    model = ViltModel.from_pretrained(cfg.model_name)

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)

    out_dir = Path(output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Saving fine-tuned model to", out_dir)
    model.save_pretrained(out_dir, safe_serialization=True)
    processor.save_pretrained(out_dir)
    print("Done.")


if __name__ == "__main__":
    export_to_hf_folder()
