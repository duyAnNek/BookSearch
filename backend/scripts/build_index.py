import torch
import pandas as pd
from PIL import Image
import os
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import faiss
import numpy as np
import json

# Bỏ qua cảnh báo từ PIL
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- CẤU HÌNH ---
# Đường dẫn CSV gốc
CSV_PATH = 'D:/my-project/Book.csv' 
# Đường dẫn model đã fine-tune (từ train.py)
MODEL_PATH = 'D:/my-project/models/fine_tuned_clip_v2' 
# Nơi lưu index và file mapping
INDEX_OUTPUT_DIR = 'models/index'
INDEX_FILE_PATH = os.path.join(INDEX_OUTPUT_DIR, 'book_image.index')
MAPPING_FILE_PATH = os.path.join(INDEX_OUTPUT_DIR, 'index_mapping.json')

# Đường dẫn cơ sở của ảnh (phần bạn muốn loại bỏ khi lưu)
# Giống với 'D:\my-project\all_covers\' trong yêu cầu
# QUAN TRỌNG: Sửa lại cho đúng với máy của bạn
# Nó dùng để chuẩn hóa path, ví dụ:
# 'D:\my-project\all_covers\img1.jpg' -> 'all_covers/img1.jpg'
BASE_IMAGE_PATH_PREFIX = 'D:/my-project/' 


def load_and_filter_data(csv_file):
    """
    Tải và lọc CSV sử dụng logic *CHÍNH XÁC* như trong BookCoverDataset
    Điều này cực kỳ quan trọng để đảm bảo index khớp với dữ liệu.
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
    
    # 2. Chuẩn bị văn bản (để lọc)
    data['title'] = data['title'].fillna('')
    data['author'] = data['author'].fillna('')
    data['description'] = data['description'].fillna('')
    data['combined_text'] = "Tựa đề: " + data['title'] + \
                             ". Tác giả: " + data['author'] + \
                             ". Mô tả: " + data['description']
    
    # 3. Bỏ qua nếu tất cả thông tin văn bản đều trống
    empty_text_mask = data['combined_text'].str.strip() == "Tựa đề: . Tác giả: . Mô tả: ."
    data = data[~empty_text_mask]

    # 4. Bỏ qua nếu file ảnh không tồn tại
    print("Checking for image file existence...")
    data['image_path'] = data['image_path'].str.replace('\\', '/')
    
    file_exists_mask = data['image_path'].apply(os.path.exists)
    data = data[file_exists_mask]
    
    print(f"FINAL valid entries for indexing: {len(data)}")

    if len(data) == 0:
        print("CẢNH BÁO: Không có dữ liệu hợp lệ nào.")
    
    return data

def main():
    if not os.path.exists(INDEX_OUTPUT_DIR):
        os.makedirs(INDEX_OUTPUT_DIR)
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Tải model và processor đã fine-tune
    print(f"Loading model from {MODEL_PATH}...")
    model = CLIPModel.from_pretrained(MODEL_PATH).to(device) # type: ignore
    processor = CLIPProcessor.from_pretrained(MODEL_PATH)
    model.eval() # Chuyển sang chế độ đánh giá (không train)

    # 2. Tải và lọc dữ liệu
    # Dùng hàm 'load_and_filter_data' ở trên để lấy dữ liệu SẠCH
    # *** CHÚ Ý: Chúng ta sẽ dùng script 'import_data.py' (Nhiệm vụ 4)
    # *** để nạp dữ liệu này vào DB.
    # *** Giờ chúng ta chỉ dùng nó để lấy danh sách ảnh.
    print("Loading and filtering data (matching training logic)...")
    filtered_data = load_and_filter_data(CSV_PATH)
    
    image_paths = filtered_data['image_path'].tolist()
    
    # Chúng ta sẽ dùng 'image_path' chuẩn hóa làm ID
    # 'D:/my-project/all_covers/img1.jpg' -> 'all_covers/img1.jpg'
    # Đây sẽ là ID chính trong DB và mapping
    
    # Chuẩn hóa đường dẫn
    def normalize_path(path):
        p = path.replace('\\', '/')
        # Xóa tiền tố
        if p.startswith(BASE_IMAGE_PATH_PREFIX):
             p = p[len(BASE_IMAGE_PATH_PREFIX):]
        return p

    normalized_paths = [normalize_path(p) for p in image_paths]
    
    # 3. Tạo embeddings cho tất cả ảnh
    all_embeddings = []
    successful_paths = [] # <-- DANH SÁCH MỚI ĐỂ LƯU PATH THÀNH CÔNG
    
    print(f"Generating embeddings for {len(image_paths)} images...")
    
    with torch.no_grad(): # Không cần tính gradient
        # Dùng zip để lặp qua cả path gốc và path đã chuẩn hóa
        for path, normalized_path in tqdm(zip(image_paths, normalized_paths), total=len(image_paths)):
            try:
                image = Image.open(path).convert('RGB')
                inputs = processor(images=image, return_tensors="pt", padding=True, truncation=True).to(device) # type: ignore
                image_features = model.get_image_features(**inputs)
                
                # Chuẩn hóa embedding
                image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
                
                # CHỈ THÊM VÀO LIST NẾU THÀNH CÔNG
                all_embeddings.append(image_features.cpu().numpy())
                successful_paths.append(normalized_path) # <-- THÊM VÀO LIST MỚI
                
            except Exception as e:
                # Bỏ qua ảnh này, không thêm vào cả hai list
                print(f"Warning: Skipping {path} due to error: {e}")
                pass 

    # Kiểm tra xem có ảnh nào được xử lý không
    if not all_embeddings:
        print("LỖI: Không thể tạo embedding cho bất kỳ ảnh nào. Dừng lại.")
        return

    # Chuyển list các array (1, 512) thành một array lớn (N, 512)
    embeddings_matrix = np.vstack(all_embeddings)
    embedding_dim = embeddings_matrix.shape[1]
    
    print(f"Created embedding matrix: {embeddings_matrix.shape}") # Sẽ là (46639, 512)

    # 4. Xây dựng Faiss Index
    # IndexFlatIP = Index cho Inner Product (tương đương Cosine Similarity khi vector đã chuẩn hóa)
    index = faiss.IndexFlatIP(embedding_dim)
    
    # Thêm embeddings vào index
    index.add(embeddings_matrix) # type: ignore
    
    print(f"Total vectors in index: {index.ntotal}") # Sẽ là 46639

    # 5. Lưu Index
    faiss.write_index(index, INDEX_FILE_PATH)
    print(f"Faiss index saved to {INDEX_FILE_PATH}")

    # 6. Tạo và lưu file mapping
    # Bây giờ, 'successful_paths' sẽ có 46639 phần tử, khớp với index.ntotal
    
    if len(successful_paths) != index.ntotal:
         # Lỗi này không nên xảy ra nữa, nhưng để đây cho an toàn
         print(f"LỖI KHÔNG MONG ĐỢI: Số lượng path ({len(successful_paths)}) không khớp" \
               f" với số lượng index ({index.ntotal}).")
         return

    # DÙNG 'successful_paths' (chỉ chứa 46639 path)
    index_to_id_map = {i: path_id for i, path_id in enumerate(successful_paths)}
    
    with open(MAPPING_FILE_PATH, 'w') as f:
        json.dump(index_to_id_map, f)
    
    print(f"Index mapping saved to {MAPPING_FILE_PATH}")
    print(f"Build index process finished. Processed {index.ntotal} valid images.")

if __name__ == "__main__":
    main()