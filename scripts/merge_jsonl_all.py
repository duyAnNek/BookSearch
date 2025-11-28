from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
ORIG_PATH = BASE_DIR / "data" / "train_image_text.jsonl"
EXTEND_PATH = BASE_DIR / "data_extend" / "train_image_text_extend.jsonl"
OUT_DIR = BASE_DIR / "data_extend"
OUT_PATH = OUT_DIR / "train_image_text_all.jsonl"


def copy_file_lines(src: Path, dst, counter_prefix: str) -> int:
    if not src.exists():
        print(f"⚠️ Không tìm thấy file: {src}")
        return 0
    count = 0
    with src.open("r", encoding="utf-8") as f_in:
        for line in f_in:
            line = line.rstrip("\n")
            if not line:
                continue
            dst.write(line + "\n")
            count += 1
    print(f"{counter_prefix}: {count} dòng")
    return count


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    total = 0
    with OUT_PATH.open("w", encoding="utf-8") as f_out:
        total += copy_file_lines(ORIG_PATH, f_out, "Sách")
        total += copy_file_lines(EXTEND_PATH, f_out, "Mở rộng")

    print(f"✅ Đã gộp tổng cộng {total} dòng vào {OUT_PATH}")


if __name__ == "__main__":
    main()
