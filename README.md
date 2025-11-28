# 📚 AI Book Search - Hệ thống tìm kiếm sách thông minh

Dự án tìm kiếm sách sử dụng AI với khả năng tìm kiếm bằng văn bản và hình ảnh, được xây dựng bằng **CLIP**, **ViLT**, **FastAPI**, **React**, và **Gemini API**.

## ✨ Tính năng

* 🔍 **Tìm kiếm đa phương thức**: Tìm kiếm bằng văn bản hoặc hình ảnh
* 🤖 **AI Models**: Hỗ trợ cả CLIP và ViLT cho embedding
* 💬 **AI Chatbot**: Tích hợp Gemini API để trả lời câu hỏi về sách
* ⚡ **Tìm kiếm nhanh**: Sử dụng Faiss index cho vector similarity search
* 🎯 **Độ chính xác cao**: Model được fine-tune trên dataset sách tiếng Việt
* 🌐 **Web Interface**: Giao diện hiện đại với React + Vite

## 🛠️ Công nghệ sử dụng

### Backend
* **FastAPI**: Web framework
* **SQLAlchemy**: ORM cho database
* **SQLite**: Database engine
* **CLIP & ViLT**: Multimodal AI models
* **Faiss**: Vector similarity search
* **Gemini API**: AI chatbot integration

### Frontend
* **React**: UI framework
* **Vite**: Build tool
* **Axios**: HTTP client

### AI/ML
* **PyTorch**: Deep learning framework
* **Transformers (HuggingFace)**: Model implementation
* **Pillow**: Image processing

## 📋 Yêu cầu hệ thống

* Python 3.11+
* Node.js 16+
* Docker & Docker Compose (tùy chọn)
* GPU với CUDA (khuyến nghị cho training)

## 🚀 Cài đặt

### Phương thức 1: Cài đặt thủ công

#### 1. Clone repository và chuẩn bị dữ liệu

```bash
git clone <repository-url>
cd my-project
```

* Đặt file `Book.csv` vào thư mục gốc
* Giải nén ảnh bìa sách vào thư mục `all_covers/`

#### 2. Cài đặt Backend

```bash
cd backend

# Tạo và kích hoạt môi trường ảo
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Cài đặt dependencies
pip install -r requirements.txt

# Import dữ liệu và build index
python scripts/import_data.py
python scripts/build_index.py

# Khởi chạy server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

#### 3. Cài đặt Frontend

```bash
cd frontend

# Cài đặt dependencies
npm install

# Khởi chạy dev server
npm run dev
```

Ứng dụng sẽ chạy tại:
- Frontend: `http://localhost:5173`
- Backend API: `http://localhost:8000`

### Phương thức 2: Docker Compose (Khuyến nghị)

```bash
docker-compose up --build
```

## 🏗️ Cấu trúc dự án

```
my-project/
├── 📁 all_covers/                         # Thư mục chứa 46,000+ ảnh bìa sách
├── 📁 crawl_data/                         # Scripts và dữ liệu crawl từ Tiki
│   ├── tiki_covers_hybrid_filelist.py     # Script crawl ảnh bìa
│   └── link_tiki.txt                      # Danh sách URLs
│
├── 📁 data/                               # Dữ liệu training và validation
│   ├── Book.csv                           # Metadata sách
│   ├── all_image_text.jsonl               # Toàn bộ dataset JSONL
│   ├── train_image_text.jsonl             # Training set
│   ├── train_image_text_small.jsonl       # Training set nhỏ (test)
│   ├── val_image_text.jsonl               # Validation set
│   ├── val_image_text_small.jsonl         # Validation set nhỏ (test)
│   └── index_vilt/                        # ViLT model index
│
├── 📁 src/                                # Source code cho training models
│   ├── config.py                          # Configuration cho training
│   ├── datasets.py                        # Dataset loaders
│   ├── train_contrastive.py               # Contrastive learning training
│   └── vilt/                              # ViLT model implementation
│       ├── __init__.py
│       ├── build_index.py                 # Build ViLT index
│       ├── export_hf.py                   # Export to HuggingFace
│       ├── infer.py                       # Inference script
│       ├── models.py                      # Model definitions
│       └── train_custom.py                # Custom training script
│
├── 📁 scripts/                            # Utility scripts
│   ├── prepare_jsonl.py                   # Chuẩn bị dữ liệu JSONL
│   ├── merge_jsonl_all.py                 # Merge JSONL files
│   ├── train_clip_jsonl.py                # Train CLIP model
│   ├── test_clip_query.py                 # Test CLIP queries
│   ├── eval_clip_vs_vilt.py               # So sánh CLIP vs ViLT
│   └── model_comparison.png               # Visualization kết quả so sánh
│
├── 📁 backend/                            # FastAPI backend
│   ├── app/
│   │   ├── main.py                        # API endpoints
│   │   ├── database.py                    # Database connection
│   │   ├── schemas.py                     # SQLAlchemy models
│   │   ├── crud.py                        # Database operations
│   │   ├── models.py                      # Pydantic models
│   │   └── services/
│   │       ├── search_service.py          # CLIP search logic
│   │       ├── vilt_search_service.py     # ViLT search logic
│   │       └── gemini_service.py          # Gemini chatbot integration
│   │
│   ├── scripts/
│   │   ├── import_data.py                 # Import CSV to database
│   │   ├── build_index.py                 # Build CLIP Faiss index
│   │   ├── build_index_vilt.py            # Build ViLT index
│   │   ├── build_vilt_mapping_vilt.py     # Build ViLT mapping
│   │   ├── train.py                       # Train CLIP model
│   │   └── analyze_training.py            # Analyze training results
│   │
│   ├── models/                            # Trained models và indexes
│   │   ├── fine_tuned_clip_v2/            # Fine-tuned CLIP model
│   │   └── index/                         # Faiss indexes
│   │       ├── book_image.index
│   │       └── index_mapping.json
│   │
│   ├── booksearch.db                      # SQLite database
│   ├── requirements.txt                   # Python dependencies
│   └── Dockerfile                         # Docker config
│
├── 📁 frontend/                           # React frontend
│   ├── src/
│   │   ├── components/
│   │   │   ├── SearchBar.jsx              # Thanh tìm kiếm
│   │   │   ├── ResultsList.jsx            # Danh sách kết quả
│   │   │   └── BookModal.jsx              # Modal chi tiết sách
│   │   ├── App.jsx                        # Root component
│   │   ├── App.css                        # Styles
│   │   ├── index.css                      # Global styles
│   │   └── main.jsx                       # Entry point
│   │
│   ├── public/                            # Static assets
│   ├── package.json                       # Node dependencies
│   ├── vite.config.js                     # Vite configuration
│   └── Dockerfile                         # Docker config
│
├── 📁 outputs/                            # Training outputs và logs
├── 📁 fine_tuned_clip_v2/                 # Backup trained models
├── 📁 img_test/                           # Test images
├── .env                                   # Environment variables
├── .gitignore                             # Git ignore rules
├── docker-compose.yml                     # Docker Compose config
├── requirements.txt                       # Root Python dependencies
└── README.md                              # File này
```

## 🔄 Quy trình làm việc

### 1. Thu thập dữ liệu (Completed ✅)

Script `crawl_data/tiki_covers_hybrid_filelist.py` crawl thông tin sách và ảnh bìa từ Tiki:
- Thu thập URLs từ `link_tiki.txt`
- Download ảnh bìa vào `all_covers/`
- Lưu metadata vào `Book.csv`

### 2. Chuẩn bị dữ liệu training (Completed ✅)

```bash
# Tạo JSONL files cho training
python scripts/prepare_jsonl.py

# Merge các files JSONL
python scripts/merge_jsonl_all.py
```

Tạo ra:
- `data/train_image_text.jsonl`: Training set (~80%)
- `data/val_image_text.jsonl`: Validation set (~20%)
- `data/all_image_text.jsonl`: Toàn bộ dataset

### 3. Training Models (Completed ✅)

#### Train CLIP Model

```bash
# Training CLIP với contrastive learning
python src/train_contrastive.py

# Hoặc sử dụng script training từ JSONL
python scripts/train_clip_jsonl.py
```

#### Train ViLT Model

```bash
# Training ViLT model
python src/vilt/train_custom.py

# Export model sang HuggingFace format
python src/vilt/export_hf.py
```

#### So sánh Models

```bash
# Đánh giá và so sánh CLIP vs ViLT
python scripts/eval_clip_vs_vilt.py
```

Kết quả lưu trong `scripts/model_comparison.csv` và `scripts/model_comparison.png`.

### 4. Import dữ liệu vào Database

```bash
cd backend
python scripts/import_data.py
```

Script này:
- Đọc `Book.csv`
- Lọc sách hợp lệ (có ảnh + text)
- Import vào SQLite database

### 5. Build Search Index

```bash
# Build CLIP Faiss index
python scripts/build_index.py

# Build ViLT index
python scripts/build_index_vilt.py
```

Tạo vector embeddings cho tất cả sách và lưu vào Faiss index.

### 6. Tìm kiếm

Khi người dùng search:

**Text Search:**
1. Query → CLIP/ViLT text encoder → embedding vector
2. Similarity search trong Faiss index
3. Lọc duplicates
4. Trả về top-k results

**Image Search:**
1. Upload image → CLIP/ViLT image encoder → embedding vector
2. Similarity search trong Faiss index
3. Lọc duplicates
4. Trả về top-k results

**Chatbot (Gemini API):**
1. User question → Gemini API
2. Context enhancement với book data
3. Generate response
4. Return answer

## 📊 Dữ liệu

* **~46,000 sách** từ Tiki
* **46,000+ ảnh bìa sách** (JPG/PNG)
* **Metadata**: title, author, description, product_url, image_path
* **Training data**: JSONL format với image-text pairs

## 🔍 API Endpoints

### Search Endpoints

#### `GET /api/search/text`
Tìm kiếm sách bằng văn bản (CLIP).

**Parameters:**
- `query` (string): Từ khóa tìm kiếm

**Response:**
```json
[
  {
    "id": "all_covers/book1.jpg",
    "image_path": "all_covers/book1.jpg",
    "title": "Tên Sách",
    "author": "Tác Giả",
    "description": "Mô tả...",
    "product_url": "https://tiki.vn/...",
    "image_url": "http://localhost:8000/images/all_covers/book1.jpg"
  }
]
```

#### `POST /api/search/image`
Tìm kiếm sách bằng hình ảnh (CLIP).

**Form Data:**
- `file`: Image file (JPG/PNG)

**Response:** Tương tự `/api/search/text`

#### `GET /api/search/vilt/text`
Tìm kiếm bằng text với ViLT model.

#### `POST /api/search/vilt/image`
Tìm kiếm bằng image với ViLT model.

### Chatbot Endpoints

#### `POST /api/chat`
Chat với Gemini AI về sách.

**Request Body:**
```json
{
  "message": "Gợi ý sách về AI"
}
```

**Response:**
```json
{
  "response": "Dựa trên dữ liệu sách, tôi gợi ý..."
}
```

### Book Endpoints

#### `GET /api/books/{book_id}`
Lấy thông tin chi tiết một cuốn sách.

#### `GET /images/{image_path}`
Serve static images.

## 🐳 Docker Deployment

### Build và Run

```bash
# Build và start tất cả services
docker-compose up --build

# Run in background
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs -f backend
docker-compose logs -f frontend
```

### Docker Services

- **backend**: FastAPI server (port 8000)
- **frontend**: React app (port 5173)

## 🔧 Configuration

### Backend Environment Variables

Tạo file `.env` trong thư mục gốc:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### Backend Settings

File `backend/app/main.py`:
- `MODEL_PATH`: Path to fine-tuned CLIP model
- `INDEX_DIR`: Path to Faiss index
- `TITLE_SIMILARITY_THRESHOLD`: 0.85 (85%)

File `backend/app/services/search_service.py`:
- `SIMILARITY_THRESHOLD`: 0.99 (99%)
- `K_MULTIPLIER`: 3

### Frontend Settings

File `frontend/src/App.jsx`:
- `API_URL`: Backend API URL (default: `http://localhost:8000`)

## 🧪 Testing

### Test CLIP Search

```bash
# Test text search
python scripts/test_clip_query.py

# Test via API
curl "http://localhost:8000/api/search/text?query=sách AI"
```

### Test Image Search

```bash
curl -X POST -F "file=@img_test/test.jpg" \
  http://localhost:8000/api/search/image
```

### Test Chatbot

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Gợi ý sách về lập trình"}'
```

### Compare Models

```bash
# So sánh performance CLIP vs ViLT
python scripts/eval_clip_vs_vilt.py
```

Kết quả lưu trong `scripts/model_comparison.csv` và `scripts/model_comparison.png`.

## 📈 Model Performance

| Model | Recall@5 | Recall@10 | Avg Similarity |
|-------|----------|-----------|----------------|
| CLIP  | ~85%     | ~92%      | 0.78          |
| ViLT  | ~82%     | ~89%      | 0.75          |

*(Số liệu ví dụ, xem `scripts/model_comparison.csv` cho kết quả thực tế)*

## 🤝 Đóng góp

Các tính năng có thể mở rộng:
- [ ] Thêm user authentication
- [ ] Implement rating & reviews
- [ ] Advanced filtering (category, price, publisher)
- [ ] Recommendation system
- [ ] Real-time updates với WebSocket
- [ ] Deploy lên cloud (AWS/GCP/Azure)

## 📝 Technical Notes

* **CLIP**: OpenAI's Contrastive Language-Image Pre-training
* **ViLT**: Vision-and-Language Transformer
* **Faiss**: Facebook AI Similarity Search (512-dim vectors)
* **Gemini API**: Google's latest LLM for chatbot
* **Embedding dimension**: 512 for CLIP, varies for ViLT
* **Similarity metric**: Cosine similarity
* **Deduplication**: Based on similarity threshold

## 🔐 Environment Setup

### Môi trường ảo Python

```bash
# Tạo môi trường cho backend
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Tạo môi trường cho AI training
python -m venv .venv_ai
.venv_ai\Scripts\activate  # Windows

# Deactivate
deactivate
```

## 📄 License

Dự án được phát triển cho mục đích giáo dục và nghiên cứu.

## 👥 Contributors

* **Nguyễn Duy An** - Developer
* **Nguyễn Quốc Huy** - Developer

---

📅 **Last Updated**: November 2025  
🔗 **Repository**: [GitHub Link]  
📧 **Contact**: [Email]
