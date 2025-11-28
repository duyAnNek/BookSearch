import csv
import json
import random
from pathlib import Path

# Đường dẫn file CSV nguồn
CSV_PATH = Path("data/Book.csv")
COL_IMAGE = 'image_url'
COL_TITLE = 'title'
COL_DESC = 'description'

# Đích JSONL
OUT_FULL = Path("data/all_image_text.jsonl")
OUT_TRAIN = Path("data/train_image_text.jsonl")
OUT_VAL = Path("data/val_image_text.jsonl")
OUT_TRAIN_SMALL = Path("data/train_image_text_small.jsonl")
OUT_VAL_SMALL = Path("data/val_image_text_small.jsonl")

def load_samples_from_csv():
    def norm_key(k: str) -> str:
        # chuẩn hóa key: bỏ BOM, bỏ ", trim khoảng trắng
        return k.strip().strip('"').lstrip('\ufeff')

    samples = []
    with CSV_PATH.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for raw_row in reader:
            # chuẩn hóa key cho từng dòng
            row = {norm_key(k): v for k, v in raw_row.items()}

            # DEBUG (chạy lần đầu): in thử keys rồi có thể comment lại
            # print(row.keys()); break

            if COL_IMAGE not in row:
                # nếu vẫn không có image_path, bỏ qua dòng này
                # hoặc print để xem thực tế key là gì
                # print("Row keys (no image_path):", row.keys())
                continue

            image = (row[COL_IMAGE] or "").strip()
            title = (row.get(COL_TITLE) or "").strip()
            desc = (row.get(COL_DESC) or "").strip()

            if not image:
                continue

            if title and desc:
                text = f"{title}. {desc}"
            elif title:
                text = title
            elif desc:
                text = desc
            else:
                continue

            samples.append({"image": image, "text": text})
    return samples

def write_jsonl(path: Path, items):
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def main():
    samples = load_samples_from_csv()
    print(f"Loaded {len(samples)} samples from CSV")

    # Shuffle để random
    random.shuffle(samples)

    # Ghi full (tùy chọn, có thể bỏ nếu không cần)
    write_jsonl(OUT_FULL, samples)

    # Chia train/val 90/10
    n = len(samples)
    n_val = int(0.1 * n)
    val_samples = samples[:n_val]
    train_samples = samples[n_val:]

    write_jsonl(OUT_TRAIN, train_samples)
    write_jsonl(OUT_VAL, val_samples)

    # Tạo file nhỏ để test pipeline (400 train / 100 val, nếu đủ số lượng)
    small_train = train_samples[: min(400, len(train_samples))]
    small_val = val_samples[: min(100, len(val_samples))]

    write_jsonl(OUT_TRAIN_SMALL, small_train)
    write_jsonl(OUT_VAL_SMALL, small_val)

    print("Done.")
    print(f"Train full: {len(train_samples)}, Val full: {len(val_samples)}")
    print(f"Train small: {len(small_train)}, Val small: {len(small_val)}")

if __name__ == "__main__":
    main()