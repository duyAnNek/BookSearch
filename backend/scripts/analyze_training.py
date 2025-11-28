"""
Script phân tích và so sánh các model đã được train
Tạo các biểu đồ matplotlib để visualize kết quả training
"""

import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend không cần GUI
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from datetime import datetime

# Bỏ qua cảnh báo từ PIL
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ModelAnalyzer:
    # --- THAY ĐỔI 1 (Giống code gốc) ---
    def __init__(self, base_model_path='D:/my-project/fine_tuned_clip_v2', device='cpu'):
        """
        Khởi tạo analyzer
        
        Args:
            base_model_path: Đường dẫn đến thư mục chứa các model đã train
            device: 'cpu' hoặc 'cuda'
        """
        self.base_model_path = base_model_path
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Tìm tất cả các epoch đã train
        self.epoch_dirs = self._find_epoch_dirs()
        print(f"Found {len(self.epoch_dirs)} trained epochs")
    
    def _find_epoch_dirs(self):
        """Tìm tất cả các thư mục epoch_* trong base_model_path"""
        epoch_dirs = []
        if not os.path.exists(self.base_model_path):
            print(f"Cảnh báo: Không tìm thấy đường dẫn: {self.base_model_path}")
            return epoch_dirs
        
        for item in os.listdir(self.base_model_path):
            item_path = os.path.join(self.base_model_path, item)
            if os.path.isdir(item_path) and item.startswith('epoch_'):
                try:
                    epoch_num = int(item.split('_')[1])
                    epoch_dirs.append((epoch_num, item_path))
                except:
                    continue
        
        # Sắp xếp theo số epoch
        epoch_dirs.sort(key=lambda x: x[0])
        return epoch_dirs
    
    # --- HÀM ĐÃ ĐƯỢC CẬP NHẬT (THAY ĐỔI LỚN) ---
    def evaluate_model_on_sample(self, model_path, csv_path, evaluation_pool_size=1000, top_k=100):
        """
        Đánh giá model trên một sample lớn và lấy metrics của top K
        
        Args:
            model_path: Đường dẫn đến model
            csv_path: Đường dẫn đến CSV
            evaluation_pool_size: Số lượng mẫu để đánh giá (ví dụ: 1000)
            top_k: Số lượng mẫu có similarity cao nhất để tính metrics (ví dụ: 100)
            
        Returns:
            metrics (dict): Các chỉ số thống kê (Mean, Median, Min, Max, Std) của Top K
            top_k_scores (np.array): Mảng chứa điểm similarity của Top K
            top_k_results (list): List các cặp Top K (image, text, similarity)
        """
        print(f"Evaluating model at {model_path}...")
        
        # Load model
        model = CLIPModel.from_pretrained(model_path).to(self.device) # type: ignore
        processor = CLIPProcessor.from_pretrained(model_path)
        model.eval()
        
        # Load và sample data
        df = pd.read_csv(csv_path)
        
        # Lọc dữ liệu hợp lệ (giống như trong train.py)
        df = df.dropna(subset=['image_path'])
        df['title'] = df['title'].fillna('')
        df['author'] = df['author'].fillna('')
        df['description'] = df['description'].fillna('')
        df['combined_text'] = "Tựa đề: " + df['title'] + \
                             ". Tác giả: " + df['author'] + \
                             ". Mô tả: " + df['description']
        
        empty_text_mask = df['combined_text'].str.strip() == "Tựa đề: . Tác giả: . Mô tả: ."
        df = df[~empty_text_mask]
        df['image_path'] = df['image_path'].str.replace('\\', '/')
        file_exists_mask = df['image_path'].apply(os.path.exists)
        df = df[file_exists_mask]
        
        # Sample (Lấy pool lớn để đánh giá)
        if len(df) > evaluation_pool_size:
            df = df.sample(n=evaluation_pool_size, random_state=42)
        else:
            # Nếu dataset nhỏ hơn pool, dùng tất cả
            print(f"  Sample pool ({evaluation_pool_size}) > dataset size ({len(df)}). Using all {len(df)} items.")

        # --- THAY ĐỔI: Thu thập tất cả kết quả thay vì chỉ scores ---
        all_results = [] 
        successful_pairs = 0
        
        with torch.no_grad():
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
                try:
                    # Load image
                    image = Image.open(row['image_path']).convert('RGB')
                    text = row['combined_text']
                    
                    # Encode
                    image_inputs = processor(images=image, return_tensors="pt", padding=True).to(self.device) # type: ignore
                    text_inputs = processor(text=text, return_tensors="pt", padding=True, max_length=77, truncation=True).to(self.device) # type: ignore
                    
                    # Get features
                    image_features = model.get_image_features(**image_inputs)
                    text_features = model.get_text_features(**text_inputs)
                    
                    # Normalize
                    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
                    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
                    
                    # Cosine similarity
                    similarity = torch.nn.functional.cosine_similarity(image_features, text_features).item()
                    
                    # Lưu lại kết quả
                    all_results.append({
                        'image_path': row['image_path'],
                        'text': text,
                        'similarity': float(similarity)
                    })
                    successful_pairs += 1

                except Exception as e:
                    continue
        
        if len(all_results) == 0:
            print("Warning: No successful pairs to evaluate.")
            return None, None, None
        
        # --- THAY ĐỔI: Sắp xếp, lấy Top K và tính metrics ---
        
        # Sắp xếp tất cả kết quả theo similarity giảm dần
        all_results.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Lấy Top K kết quả
        top_k_results = all_results[:top_k]
        
        if len(top_k_results) == 0:
            print("Warning: No results to calculate Top-K metrics (top_k=0 or all_results is empty).")
            return None, None, None
            
        # Lấy điểm số của Top K
        top_k_scores = np.array([item['similarity'] for item in top_k_results])
        
        # Tính toán metrics DỰA TRÊN TOP K SCORES
        metrics = {
            'mean_similarity': float(np.mean(top_k_scores)),
            'std_similarity': float(np.std(top_k_scores)),
            'min_similarity': float(np.min(top_k_scores)), # Min của Top K (tức là điểm của item thứ K)
            'max_similarity': float(np.max(top_k_scores)), # Max của Top K (tức là điểm cao nhất)
            'median_similarity': float(np.median(top_k_scores)), # Median của Top K
            'successful_pairs': successful_pairs, # Tổng số cặp xử lý thành công trong pool
            'total_pairs_evaluated': len(df), # Tổng số cặp trong pool
            'top_k_count': len(top_k_scores) # Số lượng thực tế (phòng trường hợp pool < top_k)
        }
        
        return metrics, top_k_scores, top_k_results
    
    # --- HÀM ĐÃ ĐƯỢC CẬP NHẬT (THAM SỐ) ---
    def compare_all_epochs(self, csv_path, evaluation_pool_size=1000, top_k=100, output_dir=None):
        """
        So sánh tất cả các epoch đã train
        
        Args:
            csv_path: Đường dẫn đến CSV
            evaluation_pool_size: Số lượng mẫu để đánh giá (ví dụ: 1000)
            top_k: Số lượng mẫu cao nhất để tính metrics (ví dụ: 100)
            output_dir: Thư mục lưu kết quả (mặc định là base_model_path)
        """
        if output_dir is None:
            output_dir = self.base_model_path
        
        if len(self.epoch_dirs) == 0:
            print("Không tìm thấy epoch nào để so sánh!")
            return
        
        print(f"\nBắt đầu đánh giá {len(self.epoch_dirs)} epochs (Pool={evaluation_pool_size}, TopK={top_k})...")
        print("=" * 60)
        
        all_metrics = []
        epoch_numbers = []
        
        # Đánh giá từng epoch
        for epoch_num, epoch_path in self.epoch_dirs:
            print(f"\nĐang đánh giá Epoch {epoch_num}...")
            
            # --- THAY ĐỔI: Truyền tham số mới ---
            metrics, _, _ = self.evaluate_model_on_sample(
                epoch_path, 
                csv_path, 
                evaluation_pool_size, 
                top_k
            )
            # --- KẾT THÚC THAY ĐỔI ---
            
            if metrics:
                metrics['epoch'] = epoch_num
                all_metrics.append(metrics)
                epoch_numbers.append(epoch_num)
                print(f"  Mean Similarity (Top {top_k}): {metrics['mean_similarity']:.4f}")
                print(f"  Std Similarity (Top {top_k}): {metrics['std_similarity']:.4f}")
        
        if len(all_metrics) == 0:
            print("Không có metrics nào để vẽ biểu đồ!")
            return
        
        # Tạo biểu đồ so sánh
        self._plot_comparison(all_metrics, output_dir)
        
        # Lưu metrics vào CSV
        df_metrics = pd.DataFrame(all_metrics)
        csv_output = os.path.join(output_dir, 'model_comparison_top_k.csv')
        df_metrics.to_csv(csv_output, index=False)
        print(f"\nTop-K Metrics saved to {csv_output}")
        
        # In summary
        self._print_summary(all_metrics)
    
    def _plot_comparison(self, metrics_list, output_dir):
        """Vẽ các biểu đồ so sánh"""
        epochs = [m['epoch'] for m in metrics_list]
        mean_sims = [m['mean_similarity'] for m in metrics_list]
        std_sims = [m['std_similarity'] for m in metrics_list]
        med_sims = [m['median_similarity'] for m in metrics_list]
        
        # Lấy Top-K count để thêm vào tiêu đề
        top_k_count = metrics_list[0]['top_k_count'] if metrics_list else 'K'
        
        # Tạo figure với nhiều subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Model Comparison - Top {top_k_count} Similarity Metrics', fontsize=16, fontweight='bold')
        
        # 1. Mean Similarity
        axes[0, 0].plot(epochs, mean_sims, 'b-o', linewidth=2, markersize=8)
        axes[0, 0].set_title('Mean Similarity Score', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Mean Similarity')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend(['Mean Similarity'])
        
        # Thêm giá trị trên điểm
        for i, (epoch, sim) in enumerate(zip(epochs, mean_sims)):
            axes[0, 0].annotate(f'{sim:.4f}', 
                                (epoch, sim), 
                                textcoords="offset points", 
                                xytext=(0,10), 
                                ha='center', 
                                fontsize=8)
        
        # 2. Standard Deviation
        axes[0, 1].plot(epochs, std_sims, 'r-s', linewidth=2, markersize=8)
        axes[0, 1].set_title('Standard Deviation of Similarity', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Std Deviation')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend(['Std Deviation'])
        
        # 3. Mean vs Median
        axes[1, 0].plot(epochs, mean_sims, 'b-o', linewidth=2, markersize=8, label='Mean')
        axes[1, 0].plot(epochs, med_sims, 'g-^', linewidth=2, markersize=8, label='Median')
        axes[1, 0].set_title('Mean vs Median Similarity', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Similarity')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        # 4. Improvement (so với epoch đầu tiên)
        if len(mean_sims) > 1:
            first_sim = mean_sims[0]
            # Đảm bảo không chia cho 0
            improvements = [(sim - first_sim) / first_sim * 100 if first_sim != 0 else 0 for sim in mean_sims]
            axes[1, 1].plot(epochs, improvements, 'm-^', linewidth=2, markersize=8)
            axes[1, 1].set_title('Improvement from First Epoch (%)', fontsize=12, fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Improvement (%)')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
            axes[1, 1].legend(['Improvement'])
            
            # Thêm giá trị improvement
            for i, (epoch, imp) in enumerate(zip(epochs, improvements)):
                axes[1, 1].annotate(f'{imp:+.2f}%', 
                                    (epoch, imp), 
                                    textcoords="offset points", 
                                    xytext=(0,10), 
                                    ha='center', 
                                    fontsize=8)
        
        plt.tight_layout()
        
        # Lưu biểu đồ
        plot_path = os.path.join(output_dir, 'model_comparison_top_k.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plots saved to {plot_path}")
        
        plt.close()
    
    def _print_summary(self, metrics_list):
        """In tóm tắt kết quả (dựa trên Top K metrics)"""
        print("\n" + "=" * 60)
        print(f"TÓM TẮT ĐÁNH GIÁ (DỰA TRÊN TOP {metrics_list[0]['top_k_count']} METRICS)")
        print("=" * 60)
        
        best_epoch = max(metrics_list, key=lambda x: x['mean_similarity'])
        worst_epoch = min(metrics_list, key=lambda x: x['mean_similarity'])
        
        print(f"\nEpoch tốt nhất: Epoch {best_epoch['epoch']}")
        print(f"  Mean Similarity: {best_epoch['mean_similarity']:.4f}")
        print(f"  Std Deviation: {best_epoch['std_similarity']:.4f}")
        
        print(f"\nEpoch kém nhất: Epoch {worst_epoch['epoch']}")
        print(f"  Mean Similarity: {worst_epoch['mean_similarity']:.4f}")
        print(f"  Std Deviation: {worst_epoch['std_similarity']:.4f}")
        
        if len(metrics_list) > 1:
            first = metrics_list[0]
            last = metrics_list[-1]
            if first['mean_similarity'] != 0:
                improvement = (last['mean_similarity'] - first['mean_similarity']) / first['mean_similarity'] * 100
                print(f"\nCải thiện từ Epoch {first['epoch']} đến Epoch {last['epoch']}: {improvement:+.2f}%")
            else:
                print(f"\nKhông thể tính cải thiện % do epoch đầu có similarity = 0")
        
        print("=" * 60)
    
    # --- HÀM ĐÃ ĐƯỢC CẬP NHẬT (THAM SỐ & LOGIC) ---
    def analyze_single_epoch(self, epoch_num, csv_path, evaluation_pool_size=1000, top_k=100, output_dir=None):
        """
        Phân tích chi tiết một epoch cụ thể
        
        Args:
            epoch_num: Số epoch cần phân tích
            csv_path: Đường dẫn đến CSV
            evaluation_pool_size: Số lượng mẫu để đánh giá
            top_k: Số lượng mẫu cao nhất để tính metrics
            output_dir: Thư mục lưu kết quả
        """
        # Tìm epoch path
        epoch_path = None
        for ep_num, ep_path in self.epoch_dirs:
            if ep_num == epoch_num:
                epoch_path = ep_path
                break
        
        if epoch_path is None:
            print(f"Không tìm thấy Epoch {epoch_num}!")
            return
        
        if output_dir is None:
            output_dir = os.path.join(self.base_model_path, f'analysis_epoch_{epoch_num}')
            os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nPhân tích chi tiết Epoch {epoch_num} (Pool={evaluation_pool_size}, TopK={top_k})...")
        
        # --- THAY ĐỔI: Lấy cả 3 giá trị trả về ---
        metrics, top_k_scores, top_k_results = self.evaluate_model_on_sample(
            epoch_path, 
            csv_path, 
            evaluation_pool_size, 
            top_k
        )
        # --- KẾT THÚC THAY ĐỔI ---
        
        if metrics:
            # --- THAY ĐỔI: Vẽ histogram của Top K scores ---
            self._plot_similarity_distribution(top_k_scores, output_dir, epoch_num)
            
            # Lưu các cặp có độ tương đồng Top K
            if top_k_results:
                top_k_path = os.path.join(output_dir, f'top_{top_k}_similarity_pairs.json')
                try:
                    # Ghi file với encoding='utf-8' để xử lý tiếng Việt
                    with open(top_k_path, 'w', encoding='utf-8') as f:
                        json.dump(top_k_results, f, indent=2, ensure_ascii=False)
                    print(f"\nĐã lưu {len(top_k_results)} cặp (Top {top_k}) vào: {top_k_path}")
                except Exception as e:
                    print(f"Lỗi khi lưu file JSON: {e}")
            
            # Lưu metrics
            with open(os.path.join(output_dir, 'top_k_metrics.json'), 'w') as f:
                json.dump(metrics, f, indent=2)
            # --- KẾT THÚC THAY ĐỔI ---
            
            print(f"\nTop-K Metrics saved to {os.path.join(output_dir, 'top_k_metrics.json')}")
            print(f"Mean Similarity: {metrics['mean_similarity']:.4f}")
            print(f"Std Deviation: {metrics['std_similarity']:.4f}")
    
    # --- HÀM ĐÃ ĐƯỢC CẬP NHẬT (TIÊU ĐỀ BIỂU ĐỒ) ---
    def _plot_similarity_distribution(self, similarities, output_dir, epoch_num):
        """Vẽ histogram phân phối similarity từ một mảng có sẵn (của Top K)"""
        print("Vẽ similarity distribution của Top K...")
        
        if similarities is None or len(similarities) == 0:
            print("Không có dữ liệu để vẽ histogram!")
            return
        
        # Vẽ histogram
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(similarities, bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(similarities), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(similarities):.4f}') # type: ignore
        ax.axvline(np.median(similarities), color='g', linestyle='--', linewidth=2, label=f'Median: {np.median(similarities):.4f}') # type: ignore
        ax.set_xlabel('Cosine Similarity', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        
        # --- THAY ĐỔI: Cập nhật tiêu đề ---
        ax.set_title(f'Top {len(similarities)} Similarity Distribution - Epoch {epoch_num}', fontsize=14, fontweight='bold')
        # --- KẾT THÚC THAY ĐỔI ---
        
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f'similarity_distribution_epoch_{epoch_num}_top_k.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Distribution plot saved to {plot_path}")
        
        plt.close()


def main():
    """Hàm main để chạy phân tích"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Phân tích và so sánh các model đã train')
    
    # --- THAY ĐỔI 2 (Giống code gốc) ---
    parser.add_argument('--base_model_path', type=str, 
                        default='D:/my-project/fine_tuned_clip_v2',
                        help='Đường dẫn đến thư mục chứa các model đã train')
    
    parser.add_argument('--csv_path', type=str,
                        default='D:/my-project/Book.csv',
                        help='Đường dẫn đến file CSV')
    
    # --- THAY ĐỔI 3: Cập nhật tham số dòng lệnh ---
    parser.add_argument('--evaluation_pool_size', type=int, default=1000,
                        help='Số lượng mẫu ngẫu nhiên để đánh giá (mặc định: 1000)')
    parser.add_argument('--top_k', type=int, default=100,
                        help='Số lượng mẫu có similarity cao nhất để tính metrics (mặc định: 100)')
    # --- KẾT THÚC THAY ĐỔI 3 ---

    parser.add_argument('--mode', type=str, choices=['compare', 'single'],
                        default='compare',
                        help='Chế độ: compare (so sánh tất cả) hoặc single (phân tích 1 epoch)')
    parser.add_argument('--epoch', type=int, default=None,
                        help='Số epoch để phân tích (chỉ dùng với mode=single)')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda'],
                        help='Device để sử dụng')
    
    args = parser.parse_args()
    
    # Xác định device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # Khởi tạo analyzer
    analyzer = ModelAnalyzer(base_model_path=args.base_model_path, device=device)
    
    if args.mode == 'compare':
        # So sánh tất cả epochs
        # --- THAY ĐỔI 4: Truyền tham số mới ---
        analyzer.compare_all_epochs(
            csv_path=args.csv_path,
            evaluation_pool_size=args.evaluation_pool_size,
            top_k=args.top_k
        )
    elif args.mode == 'single':
        if args.epoch is None:
            print("Lỗi: Cần chỉ định --epoch khi dùng mode=single")
            return
        # Phân tích một epoch
        # --- THAY ĐỔI 5: Truyền tham số mới ---
        analyzer.analyze_single_epoch(
            epoch_num=args.epoch,
            csv_path=args.csv_path,
            evaluation_pool_size=args.evaluation_pool_size,
            top_k=args.top_k
        )


if __name__ == "__main__":
    main()