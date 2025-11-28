# Sử dụng image Python 3.11
FROM python:3.11-slim

# Đặt thư mục làm việc bên trong container là /app
WORKDIR /app

# Cài đặt các thư viện hệ thống (nếu cần, ví dụ: build một số lib)
# RUN apt-get update && apt-get install -y build-essential

# Sao chép file requirements.txt vào trước
COPY requirements.txt .

# Cài đặt các thư viện Python
# --no-cache-dir để giảm kích thước image
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép toàn bộ code trong thư mục 'backend' vào /app
COPY . .

# Mở port 8000 của container
EXPOSE 8000

# Lệnh để chạy ứng dụng khi container khởi động
# Dùng "0.0.0.0" để cho phép truy cập từ bên ngoài container
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]