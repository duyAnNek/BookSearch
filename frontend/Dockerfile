# --- Giai đoạn 1: Build (Xây dựng) ---
# Sử dụng image Node.js (phiên bản 20)
FROM node:20-alpine AS build

# Đặt thư mục làm việc
WORKDIR /app

# Sao chép package.json và package-lock.json
COPY package.json package-lock.json ./

# Cài đặt thư viện
RUN npm install

# Sao chép toàn bộ code frontend
COPY . .

# Build ứng dụng
RUN npm run build

# --- Giai đoạn 2: Serve (Phục vụ) ---
# Sử dụng image Nginx
FROM nginx:alpine

# Sao chép các file đã build (từ /app/dist) vào thư mục web của Nginx
COPY --from=build /app/dist /usr/share/nginx/html

# Sao chép file config Nginx (nếu cần, nhưng mặc định là đủ)
# COPY nginx.conf /etc/nginx/conf.d/default.conf

# Mở port 80 (port mặc định của Nginx)
EXPOSE 80