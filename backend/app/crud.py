from sqlalchemy.orm import Session
from . import schemas


def get_book_by_id(db: Session, book_id: str):
    """Lấy sách bằng ID (chính là normalized_image_path)."""
    return db.query(schemas.Book).filter(schemas.Book.id == book_id).first()


def get_books_by_ids(db: Session, book_ids: list[str]):
    """Lấy danh sách sách theo list các ID."""
    return db.query(schemas.Book).filter(schemas.Book.id.in_(book_ids)).all()


def search_books_autocomplete(db: Session, query: str, limit: int = 10):
    """Autocomplete sách theo title (dùng LIKE, giới hạn số kết quả)."""
    pattern = f"%{query}%"
    return (
        db.query(schemas.Book)
        .filter(schemas.Book.title.ilike(pattern))
        .order_by(schemas.Book.title)
        .limit(limit)
        .all()
    )


def create_book(db: Session, book_data: dict):
    """Tạo một record sách mới."""
    # Tạo ID chuẩn hóa từ image_path
    path = book_data.get("image_path", "").replace("\\", "/")

    # Xóa tiền tố 'D:/my-project/' (Phải khớp với build_index.py)
    prefix = "D:/my-project/"
    if path.startswith(prefix):
        path = path[len(prefix) :]

    if not path:
        return None  # Bỏ qua nếu không có path

    db_book = schemas.Book(
        id=path,  # ID chính là path đã chuẩn hóa
        image_path=path,
        image_url=book_data.get("image_url"),
        title=book_data.get("title"),
        product_url=book_data.get("product_url"),
        author=book_data.get("author"),
        description=book_data.get("description"),
    )
    db.add(db_book)
    return db_book