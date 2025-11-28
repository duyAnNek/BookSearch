import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import os
from tqdm import tqdm

# Bỏ qua cảnh báo từ PIL khi mở ảnh có thể bị cắt bớt
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class BookCoverDataset(Dataset):
    def __init__(self, csv_file, processor):
        self.processor = processor
        
        print(f"Loading CSV from {csv_file}...")
        try:
            self.data = pd.read_csv(csv_file)
        except FileNotFoundError:
            print(f"LỖI: Không tìm thấy file CSV tại: {csv_file}")
            raise
        
        print(f"Loaded {len(self.data)} raw entries.")
        
        # --- Lọc dữ liệu (Yêu cầu 1) ---

        # 1. Bỏ qua nếu không có đường dẫn ảnh
        self.data = self.data.dropna(subset=['image_path'])
        print(f"Entries after dropping missing image_path: {len(self.data)}")

        # 2. Chuẩn bị văn bản (Yêu cầu 2)
        # Điền NaN bằng chuỗi rỗng
        self.data['title'] = self.data['title'].fillna('')
        self.data['author'] = self.data['author'].fillna('')
        self.data['description'] = self.data['description'].fillna('')
        
        # Tạo văn bản mô tả kết hợp
        self.data['combined_text'] = "Tựa đề: " + self.data['title'] + \
                                     ". Tác giả: " + self.data['author'] + \
                                     ". Mô tả: " + self.data['description']
        
        # 3. Bỏ qua nếu tất cả thông tin văn bản đều trống
        empty_text_mask = self.data['combined_text'].str.strip() == "Tựa đề: . Tác giả: . Mô tả: ."
        self.data = self.data[~empty_text_mask]
        print(f"Entries after dropping empty text: {len(self.data)}")

        # 4. Bỏ qua nếu file ảnh không tồn tại (quan trọng!)
        print("Checking for image file existence... (việc này có thể mất vài phút)")
        # Chuyển đổi đường dẫn Windows (nếu có) thành đường dẫn chuẩn
        self.data['image_path'] = self.data['image_path'].str.replace('\\', '/')
        
        # Áp dụng kiểm tra os.path.exists
        file_exists_mask = self.data['image_path'].apply(os.path.exists)
        self.data = self.data[file_exists_mask]
        print(f"FINAL valid entries after checking files: {len(self.data)}")

        if len(self.data) == 0:
            print("CẢNH BÁO: Không còn dữ liệu hợp lệ nào sau khi lọc. Vui lòng kiểm tra lại 'Book.csv' và đường dẫn ảnh.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        row = self.data.iloc[idx]
        image_path = row['image_path']
        combined_text = row['combined_text']
        
        try:
            # Load và process ảnh
            image = Image.open(image_path).convert('RGB')
            image_processed = self.processor(
                images=image,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            
            # Process văn bản với độ dài cố định
            text_processed = self.processor(
                text=combined_text,
                return_tensors="pt",
                padding="max_length",
                max_length=77,  # Độ dài context tối đa của CLIP
                truncation=True
            )
            
            return {
                'pixel_values': image_processed['pixel_values'].squeeze(0),  # Bỏ chiều batch
                'input_ids': text_processed['input_ids'].squeeze(0),        # Bỏ chiều batch
                'attention_mask': text_processed['attention_mask'].squeeze(0) # Bỏ chiều batch
            }
        
        except Exception as e:
            # Bỏ qua ảnh bị hỏng hoặc lỗi không đọc được
            print(f"Warning: Skipping file {image_path} due to error: {e}")
            return None # Sẽ được lọc bởi collate_fn

class BookSearchTrainer:
    def __init__(self, model_name='openai/clip-vit-base-patch32', device='cpu'):
        self.device = torch.device(device)
        print(f"Using device: {self.device}")
        
        # Sử dụng mô hình CLIP
        model_id = model_name
        self.model = CLIPModel.from_pretrained(model_id).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_id)
        
        # Optimizer
        self.optimizer = AdamW(self.model.parameters(), lr=5e-6) # Learning rate nhỏ cho fine-tuning

    def train_one_epoch(self, dataloader):
        self.model.train() # Đặt mô hình ở chế độ train
        running_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc="Epoch Training")
        
        for batch in progress_bar:
            # Bỏ qua các batch rỗng (do collate_fn lọc)
            if batch is None:
                continue

            # Chuyển tensors sang device
            pixel_values = batch['pixel_values'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)

            # Xóa gradients cũ
            self.optimizer.zero_grad()
            
            # --- ĐÂY LÀ PHẦN SỬA LỖI QUAN TRỌNG ---
            # Forward pass: đưa cả ảnh và văn bản vào
            # Mô hình sẽ tự tính toán contrastive loss
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                return_loss=True  # Yêu cầu mô hình trả về loss
            )
            
            # Lấy loss và thực hiện backward pass
            loss = outputs.loss
            loss.backward()
            
            # Cập nhật trọng số
            self.optimizer.step()
            
            running_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        return running_loss / len(dataloader)

    def save_model(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.model.save_pretrained(output_dir)
        self.processor.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")

def main():
    # Configuration
    # SỬA ĐƯỜNG DẪN CSV THEO YÊU CẦU CỦA BẠN
    CSV_PATH = 'D:/my-project/Book.csv'
    OUTPUT_DIR = 'D:/my-project/models/fine_tuned_clip_v2'
    BATCH_SIZE = 16  # Giữ batch size nhỏ nếu VRAM thấp
    NUM_EPOCHS = 10
    
    print("Initializing trainer...")
    trainer = BookSearchTrainer(device='cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Creating dataset...")
    dataset = BookCoverDataset(CSV_PATH, trainer.processor)
    
    if len(dataset) == 0:
        print("LỖI: Dataset rỗng. Không thể training. Thoát.")
        return

    def collate_fn(batch):
        # Lọc bỏ các giá trị None (từ ảnh hỏng)
        batch = [item for item in batch if item is not None]
        if len(batch) == 0:
            return None # Trả về None nếu cả batch đều hỏng
        
        # Stack các tensor từ các item hợp lệ
        return {
            'pixel_values': torch.stack([item['pixel_values'] for item in batch]),
            'input_ids': torch.stack([item['input_ids'] for item in batch]),
            'attention_mask': torch.stack([item['attention_mask'] for item in batch])
        }

    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=0,  # Đặt là 0 để tránh lỗi multiprocessing trên Windows
        collate_fn=collate_fn
    )
    
    print(f"Starting training for {NUM_EPOCHS} epochs...")
    for epoch in range(NUM_EPOCHS):
        print(f"--- Epoch {epoch + 1}/{NUM_EPOCHS} ---")
        avg_loss = trainer.train_one_epoch(dataloader)
        print(f"Epoch {epoch + 1} completed. Average Loss: {avg_loss:.4f}")
        
        # Lưu model sau mỗi epoch
        epoch_output_dir = os.path.join(OUTPUT_DIR, f"epoch_{epoch+1}")
        trainer.save_model(epoch_output_dir)

    print("Training finished.")
    trainer.save_model(OUTPUT_DIR)

if __name__ == "__main__":
    main()