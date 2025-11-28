from dataclasses import dataclass

@dataclass
class TrainConfig:
    # Model ViLT
    model_name: str = "dandelin/vilt-b32-mlm"

    # Dùng tập nhỏ để test trước
    train_file: str = "data/train_image_text.jsonl"
    val_file: str = "data/val_image_text.jsonl"

    # Ảnh đang là URL nên tạm không dùng image_root
    batch_size: int = 4          # RTX 4060, an toàn
    lr: float = 1e-6
    num_epochs: int = 1         
    output_dir: str = "outputs/checkpoints/vilt_vi"

cfg = TrainConfig()