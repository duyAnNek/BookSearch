from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from typing import List
from difflib import SequenceMatcher  # <-- THÊM THƯ VIỆN NÀY

from . import crud, models, schemas
from .database import get_db, engine
from .services.search_service import SearchService
from .services.vilt_search_service import ViltSearchService
from .services.gemini_service import GeminiSearchSuggestor

# --- CẤU HÌNH ---
# Model CLIP CŨ đã dùng ổn định trong backend (fine_tuned_clip_v2)
MODEL_PATH = "models/fine_tuned_clip_v2"
# Index CŨ tương ứng với model cũ
INDEX_DIR = "models/index"
VILT_INDEX_DIR = "/data_root/data/index_vilt"

# Ngưỡng tương đồng cho TÊN SÁCH (TITLE)
# 0.85 = 85% giống nhau. Bạn có thể điều chỉnh con số này
TITLE_SIMILARITY_THRESHOLD = 0.95

# Tạo database tables
schemas.Base.metadata.create_all(bind=engine)

# Khởi tạo FastAPI app
app = FastAPI(title="Book Search API")

# --- Cài đặt CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Khởi tạo Service Tìm kiếm (Singleton) ---
search_service = SearchService(model_path=MODEL_PATH, index_dir=INDEX_DIR)
vilt_search_service = ViltSearchService(index_dir=VILT_INDEX_DIR)
gemini_suggestor = GeminiSearchSuggestor()

# --- Mount Static Files (Để phục vụ ảnh) ---
app.mount("/images", StaticFiles(directory="/data_root"), name="images")


# === HÀM LỌC TÊN TƯƠNG ĐỒNG (MỚI) ===
def filter_similar_titles(books: List[schemas.Book]) -> List[schemas.Book]:
    """
    Lọc một danh sách sách, loại bỏ những cuốn có tên quá giống nhau.
    """
    final_results = []
    kept_titles = [] # Lưu các title đã được giữ lại

    for book in books:
        if not book.title: # type: ignore # Bỏ qua nếu sách không có têncontinue
            continue

        current_title = book.title.lower() # Chuẩn hóa về chữ thường
        is_duplicate = False

        for kept_title in kept_titles:
            # Tính toán độ tương đồng
            similarity = SequenceMatcher(None, current_title, kept_title).ratio()
            
            if similarity > TITLE_SIMILARITY_THRESHOLD:
                is_duplicate = True
                break # Tìm thấy trùng lặp, không cần kiểm tra thêm
        
        if not is_duplicate:
            final_results.append(book)
            kept_titles.append(current_title)
            
    return final_results

# === API Endpoints (ĐÃ CẬP NHẬT) ===

@app.get("/api/autocomplete", response_model=List[models.BookSearchResult])
def autocomplete(
    query: str = Query(..., min_length=1, max_length=100),
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db),
):
    """Gợi ý sách nhanh theo title (autocomplete, chỉ dùng DB)."""
    try:
        books = crud.search_books_autocomplete(db, query=query, limit=limit)
        # map sang BookSearchResult
        return [
            models.BookSearchResult(
                id=b.id,
                image_path=b.image_path,
                title=b.title,
                author=b.author,
            )
            for b in books
        ]
    except Exception as e:
        print(f"Lỗi autocomplete: {e}")
        raise HTTPException(status_code=500, detail="Lỗi máy chủ nội bộ")

@app.get("/api/autocomplete/gemini", response_model=List[str])
def autocomplete_gemini(
    query: str = Query(..., min_length=1, max_length=100),
):
    """Gợi ý câu truy vấn tìm sách bằng Gemini (không dùng DB)."""
    try:
        suggestions = gemini_suggestor.suggest_queries(query, max_suggestions=5)
        return suggestions
    except Exception as e:
        print(f"Lỗi autocomplete_gemini: {e}")
        raise HTTPException(status_code=500, detail="Lỗi gợi ý Gemini")

@app.get("/api/search/text", response_model=List[models.BookDetail])
def search_text(
    query: str = Query(..., min_length=1, max_length=100),
    db: Session = Depends(get_db)
):
    """
    Tìm kiếm sách bằng truy vấn văn bản
    """
    try:
        # 1. Tìm kiếm bằng AI (đã lọc trùng lặp HÌNH ẢNH)
        # Chúng ta lấy nhiều hơn (top_k=50) để có dữ liệu cho việc lọc TÊN
        search_results = search_service.search_by_text(query, top_k=50) 
        
        if not search_results:
            return []
            
        book_ids = [book_id for book_id, score in search_results]
        
        # 2. Truy vấn DB
        books_from_db = crud.get_books_by_ids(db, book_ids)
        
        # 3. Sắp xếp lại theo thứ tự của AI
        book_map = {book.id: book for book in books_from_db}
        ordered_books = [book_map[book_id] for book_id in book_ids if book_id in book_map]
        
        # 4. === LỌC TÊN TƯƠNG ĐỒNG (MỚI) ===
        final_filtered_books = filter_similar_titles(ordered_books)
        
        return final_filtered_books
        
    except Exception as e:
        print(f"Lỗi tìm kiếm văn bản: {e}")
        raise HTTPException(status_code=500, detail="Lỗi máy chủ nội bộ")

@app.get("/api/vilt/search/text", response_model=List[models.ViltSearchResult])
def vilt_search_text(
    query: str = Query(..., min_length=1, max_length=200),
):
    """Tìm kiếm sách bằng ViLT (không phụ thuộc DB, dựa trên JSONL/index)."""
    try:
        results = vilt_search_service.search_by_text(query, top_k=20)
        return results
    except Exception as e:
        print(f"Lỗi tìm kiếm ViLT: {e}")
        raise HTTPException(status_code=500, detail="Lỗi máy chủ nội bộ")

@app.post("/api/search/image", response_model=List[models.BookDetail])
async def search_image(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Tìm kiếm sách bằng hình ảnh tải lên
    """
    try:
        image_bytes = await file.read()
        
        # 1. Tìm kiếm bằng AI (đã lọc trùng lặp HÌNH ẢNH)
        search_results = search_service.search_by_image(image_bytes, top_k=50)
        
        if not search_results:
            return []
            
        book_ids = [book_id for book_id, score in search_results]
        
        # 2. Truy vấn DB
        books_from_db = crud.get_books_by_ids(db, book_ids)
        
        # 3. Sắp xếp lại
        book_map = {book.id: book for book in books_from_db}
        ordered_books = [book_map[book_id] for book_id in book_ids if book_id in book_map]
        
        # 4. === LỌC TÊN TƯƠNG ĐỒNG (MỚI) ===
        final_filtered_books = filter_similar_titles(ordered_books)

        return final_filtered_books

    except Exception as e:
        print(f"Lỗi tìm kiếm ảnh: {e}")
        raise HTTPException(status_code=500, detail="Lỗi máy chủ nội bộ")

@app.get("/api/books/{book_id:path}", response_model=models.BookDetail)
def get_book_details(book_id: str, db: Session = Depends(get_db)):
    """
    Lấy thông tin chi tiết của một cuốn sách bằng ID
    """
    book = crud.get_book_by_id(db, book_id)
    if book is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy sách")
    return book

@app.get("/")
def read_root():
    return {"message": "Chào mừng đến với API Tìm kiếm Sách!"}