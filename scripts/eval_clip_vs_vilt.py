import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List

import json
import matplotlib.pyplot as plt
import requests

API_BASE = "http://localhost:8000"
JSONL_PATH = Path("data/val_image_text.jsonl")


@dataclass
class EvalQuery:
    id: int
    query: str
    keywords: List[str]


def build_eval_queries_from_jsonl(max_queries: int = 1000) -> List[EvalQuery]:
    """Sinh EvalQuery từ file JSONL nhỏ.

    - query: câu đầu tiên trong trường text.
    - keywords: một vài từ khóa dài (>=4 ký tự) xuất hiện trong text.
    """

    queries: List[EvalQuery] = []
    if not JSONL_PATH.exists():
        return queries

    with JSONL_PATH.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = obj.get("text", "").strip()
            if not text:
                continue

            # câu đầu tiên làm query (đến dấu chấm đầu tiên), cắt tối đa 1000 ký tự
            first_sentence = text.split(".")[0]
            query = first_sentence.strip()
            if not query:
                query = text

            query = query[:1000]

            # trích vài keyword đơn giản: các từ dài, bỏ trùng
            tokens = [t.strip(',.:;!?()"').lower() for t in text.split()]
            tokens = [t for t in tokens if len(t) >= 4]
            seen = set()
            keywords: List[str] = []
            for t in tokens:
                if t not in seen:
                    seen.add(t)
                    keywords.append(t)
                if len(keywords) >= 5:
                    break

            if not keywords:
                continue

            queries.append(EvalQuery(i, query, keywords))

            if len(queries) >= max_queries:
                break

    return queries


def call_api(path: str, params: dict) -> List[dict]:
    url = f"{API_BASE}{path}"

    # Đảm bảo không vượt quá giới hạn max_length=100 của tham số query bên backend
    if "query" in params and isinstance(params["query"], str):
        safe_params = dict(params)
        safe_params["query"] = safe_params["query"][:100]
    else:
        safe_params = params

    resp = requests.get(url, params=safe_params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _contains_any(text: str, keywords: List[str]) -> bool:
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in keywords)


def evaluate_model(model_name: str, path: str, queries: List[EvalQuery], max_k: int = 10) -> dict:
    """Trả về dict: {"hit@1": float, "hit@5": float, "hit@10": float}."""

    hits_at_1 = 0
    hits_at_5 = 0
    hits_at_10 = 0

    for q in queries:
        results = call_api(path, {"query": q.query})

        # Lấy text để so với keywords
        titles: List[str] = []
        for item in results[:max_k]:
            # CLIP: BookDetail có field 'title'
            # ViLT: ViltSearchResult có thể có 'text' hoặc 'title'
            title = item.get("title") or item.get("text") or ""
            titles.append(title)

        # hit@1
        if titles:
            if _contains_any(titles[0], q.keywords):
                hits_at_1 += 1

        # hit@5
        if any(_contains_any(t, q.keywords) for t in titles[:5]):
            hits_at_5 += 1

        # hit@10
        if any(_contains_any(t, q.keywords) for t in titles[:10]):
            hits_at_10 += 1

    n = len(queries)
    return {
        "hit@1": hits_at_1 / n,
        "hit@5": hits_at_5 / n,
        "hit@10": hits_at_10 / n,
    }


def save_results_csv(results: dict, out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "metric", "value"])
        for model_name, metrics in results.items():
            for metric_name, value in metrics.items():
                writer.writerow([model_name, metric_name, value])


def plot_results(results: dict, out_path: Path) -> None:
    models = list(results.keys())
    metrics = ["hit@1", "hit@5", "hit@10"]

    x = range(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, metric in enumerate(metrics):
        vals = [results[m][metric] for m in models]
        ax.bar([xx + i * width for xx in x], vals, width, label=metric)

    ax.set_xticks([xx + width / 2 for xx in x])
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title("So sánh CLIP vs ViLT (hit@k dựa trên keywords)")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path)


def main() -> None:
    # Đảm bảo backend đang chạy trên http://localhost:8000
    queries = build_eval_queries_from_jsonl(max_queries=1000)
    if not queries:
        print("Không tạo được EvalQuery nào từ", JSONL_PATH)
        return

    print(f"Số query đánh giá: {len(queries)}")

    print("Đánh giá CLIP ( /api/search/text ) ...")
    clip_metrics = evaluate_model("CLIP", "/api/search/text", queries)

    print("Đánh giá ViLT ( /api/vilt/search/text ) ...")
    vilt_metrics = evaluate_model("ViLT", "/api/vilt/search/text", queries)

    results = {
        "CLIP": clip_metrics,
        "ViLT": vilt_metrics,
    }

    out_dir = Path("scripts")
    out_dir.mkdir(exist_ok=True)

    csv_path = out_dir / "model_comparison.csv"
    png_path = out_dir / "model_comparison.png"

    save_results_csv(results, csv_path)
    plot_results(results, png_path)

    print("Kết quả:")
    for model_name, metrics in results.items():
        print(f"  {model_name}:")
        for metric_name, value in metrics.items():
            print(f"    {metric_name}: {value:.3f}")

    print("Đã lưu:")
    print("  ", csv_path)
    print("  ", png_path)


if __name__ == "__main__":
    main()
