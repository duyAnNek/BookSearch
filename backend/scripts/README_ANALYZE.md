# Hướng dẫn sử dụng Script Phân tích Model

Script `analyze_training.py` cho phép bạn phân tích và so sánh các model đã được train từ các epoch khác nhau.

## Cài đặt

Cài đặt matplotlib (nếu chưa có):
```bash
pip install matplotlib
```

## Cách sử dụng

### 1. So sánh tất cả các epochs

So sánh tất cả các model đã train để tìm epoch tốt nhất:

```bash
cd backend
python scripts/analyze_training.py --mode compare --csv_path "D:/my-project/Book.csv" --sample_size 100
```

**Kết quả:**
- File `model_comparison.png`: Biểu đồ so sánh 4 metrics
- File `model_comparison.csv`: Bảng dữ liệu metrics của tất cả epochs

### 2. Phân tích chi tiết một epoch cụ thể

Phân tích sâu một epoch để xem phân phối similarity:

```bash
python scripts/analyze_training.py --mode single --epoch 5 --csv_path "D:/my-project/Book.csv" --sample_size 200
```

**Kết quả:**
- File `similarity_distribution_epoch_5.png`: Histogram phân phối similarity
- File `metrics.json`: Các metrics chi tiết

### 3. Tùy chọn nâng cao

```bash
# Chỉ định device cụ thể
python scripts/analyze_training.py --mode compare --device cuda

# Thay đổi đường dẫn model
python scripts/analyze_training.py --mode compare --base_model_path "D:/my-project/models/custom_path"

# Tăng số lượng mẫu test (chính xác hơn nhưng chậm hơn)
python scripts/analyze_training.py --mode compare --sample_size 500
```

## Các tham số

| Tham số | Mô tả | Mặc định |
|---------|-------|----------|
| `--base_model_path` | Đường dẫn đến thư mục chứa models | `D:/my-project/models/fine_tuned_clip_v2` |
| `--csv_path` | Đường dẫn đến file CSV | `D:/my-project/Book.csv` |
| `--sample_size` | Số lượng mẫu để test | `100` |
| `--mode` | Chế độ: `compare` hoặc `single` | `compare` |
| `--epoch` | Số epoch (chỉ dùng với `--mode single`) | `None` |
| `--device` | Device: `auto`, `cpu`, hoặc `cuda` | `auto` |

## Các metrics được tính toán

1. **Mean Similarity**: Độ tương đồng trung bình giữa ảnh và text
2. **Std Deviation**: Độ lệch chuẩn của similarity
3. **Min/Max Similarity**: Giá trị nhỏ nhất/lớn nhất
4. **Median Similarity**: Giá trị trung vị

## Các biểu đồ được tạo

### Mode `compare`:
1. **Mean Similarity Score**: Độ tương đồng trung bình qua các epochs
2. **Standard Deviation**: Độ lệch chuẩn qua các epochs
3. **Mean vs Median**: So sánh mean và median
4. **Improvement**: Phần trăm cải thiện so với epoch đầu tiên

### Mode `single`:
- **Similarity Distribution**: Histogram phân phối similarity với mean và median

## Ví dụ output

Sau khi chạy, bạn sẽ thấy:

```
Using device: cuda
Found 10 trained epochs

Bắt đầu đánh giá 10 epochs...
============================================================

Đang đánh giá Epoch 1...
Processing: 100%|████████████| 100/100
  Mean Similarity: 0.8234
  Std Similarity: 0.0542

...

TÓM TẮT ĐÁNH GIÁ
============================================================

Epoch tốt nhất: Epoch 10
  Mean Similarity: 0.9123
  Std Deviation: 0.0432

Epoch kém nhất: Epoch 1
  Mean Similarity: 0.8234
  Std Deviation: 0.0542

Cải thiện từ Epoch 1 đến Epoch 10: +10.79%
============================================================
```

## Lưu ý

- Script sẽ tự động tìm tất cả các thư mục `epoch_*` trong `base_model_path`
- Sử dụng `--sample_size` lớn hơn sẽ cho kết quả chính xác hơn nhưng mất nhiều thời gian hơn
- Nếu bạn có GPU, script sẽ tự động sử dụng GPU để tăng tốc
- Các file kết quả sẽ được lưu trong thư mục `base_model_path`


