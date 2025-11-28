import torch
import faiss
import json
import os
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
from io import BytesIO

class SearchService:
    def __init__(self, model_path: str, index_dir: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = model_path
        self.index_file = os.path.join(index_dir, 'book_image.index')
        self.mapping_file = os.path.join(index_dir, 'index_mapping.json')
        
        self.model = None
        self.processor = None
        self.index = None
        self.mapping = None
        
        # --- THÊM CÁC HẰNG SỐ CẤU HÌNH ---
        # Ngưỡng tương đồng để coi là trùng lặp.
        # 0.95 = 95% giống nhau. Bạn có thể tăng/giảm giá trị này.
        self.SIMILARITY_THRESHOLD = 0.99     

        # Lấy gấp 4 lần số lượng yêu cầu để có dữ liệu lọc
        self.K_MULTIPLIER = 10

        self._load_resources()

    def _load_resources(self):
        """Tải model, processor, index, và mapping vào bộ nhớ."""
        print("Loading search resources...")
        
        # Tải Model & Processor
        print(f"Loading model from {self.model_path}...")
        self.model = CLIPModel.from_pretrained(self.model_path).to(self.device) # type: ignore
        self.processor = CLIPProcessor.from_pretrained(self.model_path)
        self.model.eval()
        
        # Tải Faiss Index
        print(f"Loading Faiss index from {self.index_file}...")
        self.index = faiss.read_index(self.index_file)
        
        # Tải Mapping (map từ int index -> book_id (str))
        print(f"Loading index mapping from {self.mapping_file}...")
        with open(self.mapping_file, 'r') as f:
            # JSON lưu key là string, cần convert về int
            self.mapping = {int(k): v for k, v in json.load(f).items()}
            
        print("Search resources loaded successfully.")

    def _embed_text(self, text: str):
        """Tạo embedding cho một chuỗi văn bản."""
        with torch.no_grad():
            inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True, max_length=77).to(self.device) # type: ignore
            
            text_features = self.model.get_text_features(**inputs) # type: ignore
            # Chuẩn hóa
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            return text_features.cpu().numpy()

    def _embed_image(self, image_bytes: bytes):
        """Tạo embedding cho một ảnh (từ bytes)."""
        with torch.no_grad():
            image = Image.open(BytesIO(image_bytes)).convert('RGB')
            inputs = self.processor(images=image, return_tensors="pt", padding=True, truncation=True).to(self.device) # type: ignore
            
            image_features = self.model.get_image_features(**inputs) # type: ignore
            # Chuẩn hóa
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            return image_features.cpu().numpy()

    # --- ĐÂY LÀ PHẦN ĐƯỢC THAY ĐỔI NHIỀU NHẤT ---
    def search(self, query_vector: np.ndarray, top_k: int = 49):
        """
        Thực hiện tìm kiếm trên Faiss index VÀ lọc kết quả trùng lặp.
        Trả về list các (book_id, similarity_score)
        """
        if self.index is None or self.mapping is None:
            raise Exception("Search service not initialized.")
            
        # 1. Lấy nhiều kết quả hơn (ví dụ top_k=49 -> k_to_fetch=49*3)
        k_to_fetch = top_k * self.K_MULTIPLIER
        
        distances, indices = self.index.search(query_vector, k_to_fetch)
        
        filtered_results = []
        kept_embeddings = [] # Lưu embeddings của các ảnh đã được giữ lại
        
        for i, idx_int in enumerate(indices[0]): # idx_int là một số nguyên (index)
            # 2. Lấy thông tin
            # Faiss thỉnh thoảng trả về -1 nếu không có đủ kết quả
            if idx_int == -1 or idx_int not in self.mapping:
                continue
                
            book_id = self.mapping[idx_int]
            score = float(distances[0][i])
            
            # 3. Lấy vector embedding gốc từ Faiss index
            # .reconstruct() lấy lại vector từ index bằng ID của nó
            current_embedding = self.index.reconstruct(int(idx_int))

            # 4. So sánh với các embedding đã giữ lại
            is_duplicate = False
            for kept_emb in kept_embeddings:
                # Tính cosine similarity (vectors đã được chuẩn hóa, nên chỉ cần dot product)
                similarity = np.dot(current_embedding, kept_emb)
                
                if similarity > self.SIMILARITY_THRESHOLD:
                    is_duplicate = True
                    break # Tìm thấy 1 cái trùng là đủ

            # 5. Quyết định giữ hay bỏ
            if not is_duplicate:
                filtered_results.append((book_id, score))
                kept_embeddings.append(current_embedding)
            
            # 6. Dừng lại khi đã đủ số lượng
            if len(filtered_results) >= top_k:
                break
        
        return filtered_results

    def search_by_text(self, text_query: str, top_k: int = 20):
        query_vector = self._embed_text(text_query)
        return self.search(query_vector, top_k) # search() giờ đã tự động lọc

    def search_by_image(self, image_bytes: bytes, top_k: int = 20):
        query_vector = self._embed_image(image_bytes)
        return self.search(query_vector, top_k) # search() giờ đã tự động lọc