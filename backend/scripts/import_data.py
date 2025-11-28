import pandas as pd
import os
from sqlalchemy.orm import Session
from tqdm import tqdm

# Import các thành phần database từ app
import sys
# Thêm thư mục 'backend' vào sys.path để import app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.database import SessionLocal, engine
from app import schemas, crud

# --- CẤU HÌNH ---
CSV_PATH = 'D:/my-project/Book.csv' # Chạy từ thư mục backend/
# Đường dẫn này phải khớp với BASE_IMAGE_PATH_PREFIX trong build_index.py
BASE_IMAGE_PATH_PREFIX = 'D:/my-project/' 


def clean_and_load_data(csv_file):
    """
    Sử dụng logic lọc tương tự train.py để đảm bảo tính nhất quán
    """
    print(f"Loading CSV from {csv_file}...")
    try:
        data = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file CSV tại: {csv_file}")
        raise
    
    print(f"Loaded {len(data)} raw entries.")

    # 1. Bỏ qua nếu không có đường dẫn ảnh
    data = data.dropna(subset=['image_path'])
    print(f"After dropping missing image_path: {len(data)}")

    # 2. Chuẩn bị văn bản (fill NaN)
    data['title'] = data['title'].fillna('')
    data['author'] = data['author'].fillna('')
    data['description'] = data['description'].fillna('')
    data['combined_text'] = "Tựa đề: " + data['title'] + \
                             ". Tác giả: " + data['author'] + \
                             ". Mô tả: " + data['description']
    
    # 3. Bỏ qua nếu tất cả thông tin văn bản đều trống
    empty_text_mask = data['combined_text'].str.strip() == "Tựa đề: . Tác giả: . Mô tả: ."
    data = data[~empty_text_mask]
    print(f"After dropping empty text: {len(data)}")

    # 4. Bỏ qua nếu file ảnh không tồn tại
    print("Checking for image file existence...")
    data['image_path'] = data['image_path'].str.replace('\\', '/')
    
    file_exists_mask = data['image_path'].apply(os.path.exists)
    data = data[file_exists_mask]
    print(f"FINAL valid entries to import: {len(data)}")

    if len(data) == 0:
        print("CẢNH BÁO: Không có dữ liệu hợp lệ nào để import.")
        return None
    
    # Chuyển đổi DataFrame thành dict để import
    # Quan trọng: Chuẩn hóa image_path ở đây
    def normalize_path(path):
        p = path.replace('\\', '/')
        if p.startswith(BASE_IMAGE_PATH_PREFIX):
             p = p[len(BASE_IMAGE_PATH_PREFIX):]
        return p

    data['normalized_path'] = data['image_path'].apply(normalize_path)
    
    # Đổi tên cột để khớp với hàm create_book
    data_to_import = data.rename(columns={'normalized_path': 'id'})
    # Cần giữ lại image_path gốc để hàm create_book xử lý
    # Hoặc sửa hàm create_book...
    # Cách đơn giản hơn:
    
    data_dict = data.to_dict('records')
    return data_dict

def main():
    print("Starting data import process...")
    
    # 1. Tạo tables trong DB (nếu chưa có)
    print("Creating database tables...")
    schemas.Base.metadata.create_all(bind=engine)
    
    # 2. Lấy session DB
    db: Session = SessionLocal()
    
    # 3. Đọc và lọc dữ liệu
    records = clean_and_load_data(CSV_PATH)
    
    if not records:
        print("Không có dữ liệu để import. Thoát.")
        db.close()
        return

    # 4. Import vào DB
    print(f"Importing {len(records)} records into the database...")
    imported_count = 0
    skipped_count = 0
    
    for record in tqdm(records):
        # ID chuẩn hóa (ví dụ: 'all_covers/img1.jpg')
        prefix = BASE_IMAGE_PATH_PREFIX
        path = record.get('image_path', '').replace('\\', '/')
        if path.startswith(prefix):
            path = path[len(prefix):]
        
        book_id = path
        
        # Kiểm tra xem sách đã tồn tại chưa
        existing_book = crud.get_book_by_id(db, book_id)
        if not existing_book:
            # Dùng lại record gốc (dict)
            book_data = {
                "image_path": record.get('image_path'), # Path gốc
                "image_url": record.get('image_url'),
                "title": record.get('title'),
                "product_url": record.get('product_url'),
                "author": record.get('author'),
                "description": record.get('description')
            }
            crud.create_book(db, book_data)
            imported_count += 1
        else:
            skipped_count += 1

    # 5. Commit và đóng
    try:
        db.commit()
        print("Data committed to database.")
    except Exception as e:
        print(f"Lỗi khi commit: {e}")
        db.rollback()
    finally:
        db.close()
        
    print(f"Import finished. {imported_count} new records added. {skipped_count} records skipped (already exist).")

if __name__ == "__main__":
    main()