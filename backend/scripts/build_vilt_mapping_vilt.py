import json
from pathlib import Path
import sys

from sqlalchemy.orm import Session

# Thêm backend vào sys.path để import được app.*
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.append(str(BACKEND_ROOT))

from app.database import SessionLocal, engine  # type: ignore
from app.schemas import Book, Base  # type: ignore


DATA_PATH = Path("data/train_image_text.jsonl")
INDEX_DIR = Path("data/index_vilt")
MAPPING_PATH = INDEX_DIR / "index_mapping_vilt.json"


def build_mapping(db: Session) -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Train JSONL not found: {DATA_PATH}")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    mapping: dict[int, str] = {}
    total = 0
    matched = 0

    with DATA_PATH.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            total += 1

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            image_url = obj.get("image")
            if not image_url:
                continue

            book = db.query(Book).filter(Book.image_url == image_url).first()
            if not book:
                continue

            mapping[idx] = book.id
            matched += 1

    with MAPPING_PATH.open("w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    print("Built ViLT index->book_id mapping:")
    print("  total JSONL lines:", total)
    print("  matched books   :", matched)
    print("  saved mapping to:", MAPPING_PATH)


def main() -> None:
    # Đảm bảo bảng books tồn tại trong SQLite
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        build_mapping(db)
    finally:
        db.close()


if __name__ == "__main__":
    main()
