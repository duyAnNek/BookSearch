#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tiki covers crawler – Bản hỗ trợ đọc FILE DANH SÁCH URL
------------------------------------------------------
• 2 chế độ: API (khuyên dùng) và Browser (Playwright) để cuộn + bắt JSON.
• Đầu ra: CSV gồm 6 cột [image_path, image_url, title, product_url, author, description].
• Tải ảnh bìa về thư mục chỉ định; tự tránh trùng tên bằng cách thêm (2), (3)...
• Tự phân tích URL danh mục dạng https://tiki.vn/<url-key>/c<id>.
• NHẬN NHIỀU URL qua: --category-url (nhiều giá trị) hoặc --category-file (mỗi dòng 1 URL).
• Khi nhiều URL: tự tách CSV/thư mục ảnh theo danh mục dưới dạng <url-key>_<cid>.csv và thư mục <url-key>_<cid>/.

Cài đặt phụ thuộc:
  pip install requests playwright
  playwright install chromium

Ví dụ chạy (PowerShell):
  python .\tiki_covers_hybrid_filelist.py `
    --category-file "D:\\my-project\\link_tiki.txt" `
    --mode api `
    --max-pages 120 `
    --sleep 0.12

Hoặc truyền nhiều URL trực tiếp:
  python .\tiki_covers_hybrid_filelist.py `
    --category-url "https://tiki.vn/sach-hoc-tieng-han/c1526" "https://tiki.vn/sach-hoc-tieng-trung/c1103" `
    --mode api `
    --max-pages 120 `
    --sleep 0.12
"""
from __future__ import annotations

import argparse
import csv
import html
import os
import re
import sys
import threading
import time
from contextlib import suppress
from typing import Dict, Iterable, List, Tuple
from urllib.parse import urljoin, urlparse

import requests

# --------------------------- Cấu hình ---------------------------
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) tiki-covers/3.2",
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8",
}

DEFAULT_TIMEOUT = 25
DEFAULT_CONNECT_TIMEOUT = 10

# ---------------------- Tiện ích chuỗi & path -------------------

def clean_text(x: str) -> str:
    """Làm sạch mô tả: bỏ HTML, gộp khoảng trắng, thay <br> thành \n."""
    if not x:
        return ""
    x = html.unescape(x)
    x = re.sub(r"<br\s*/?>", "\n", x, flags=re.I)
    x = re.sub(r"<[^>]+>", " ", x)
    x = re.sub(r"\s+", " ", x)
    return x.strip()


def clean(s: str) -> str:
    if not s:
        return ""
    return re.sub(r"\s+", " ", s).strip()


def safe_filename(s: str, maxlen: int = 140) -> str:
    s = (s or "book").strip()
    s = re.sub(r'[\\/*?:"<>|]', "_", s)
    s = re.sub(r"\s+", " ", s)
    return s[:maxlen] if s else "book"


def _abs_unix_path(p: str) -> str:
    """Trả đường dẫn tuyệt đối dùng dấu / để Excel và các app đọc chuẩn."""
    return os.path.abspath(p).replace("\\", "/")


def _unique_path(base_path: str) -> str:
    """Nếu file đã tồn tại thì thêm (2), (3), ..."""
    if not os.path.exists(base_path):
        return base_path
    root, ext = os.path.splitext(base_path)
    i = 2
    while True:
        cand = f"{root} ({i}){ext}"
        if not os.path.exists(cand):
            return cand
        i += 1


def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)


# --------------------------- HTTP helpers -----------------------

def _make_session() -> requests.Session:
    """Tạo session có retry nhẹ nhàng."""
    from requests.adapters import HTTPAdapter
    try:
        from urllib3.util.retry import Retry  # type: ignore
    except Exception:  # Fallback nếu môi trường thiếu urllib3 Retry
        s = requests.Session()
        s.headers.update(HEADERS)
        return s

    retry = Retry(
        total=3,
        read=3,
        connect=3,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "HEAD"),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=50)
    s = requests.Session()
    s.headers.update(HEADERS)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s

SESSION = _make_session()


def http_get_json(url: str, *, params: Dict | None = None) -> Dict:
    r = SESSION.get(url, params=params, timeout=(DEFAULT_CONNECT_TIMEOUT, DEFAULT_TIMEOUT))
    r.raise_for_status()
    return r.json()  # type: ignore[return-value]


def http_get_bytes(url: str) -> bytes:
    r = SESSION.get(url, timeout=(DEFAULT_CONNECT_TIMEOUT, DEFAULT_TIMEOUT))
    r.raise_for_status()
    return r.content


# ----------------------- Tải ảnh bìa ----------------------------

def download_image(url: str, out_dir: str, title: str) -> str:
    """
    - Tạo thư mục (nếu thiếu)
    - Đặt tên file an toàn
    - Tránh đè file (thêm (2), (3)...)
    - Trả về đường dẫn tuyệt đối (slash '/')
    """
    if not url:
        return ""
    ensure_dir(out_dir)

    path_part = urlparse(url).path
    ext = os.path.splitext(path_part)[1] or ".jpg"
    fname = f"{safe_filename(title)}{ext}"
    base = os.path.join(out_dir, fname)
    final_path = _unique_path(base)

    with suppress(Exception):
        content = http_get_bytes(url)
        with open(final_path, "wb") as f:
            f.write(content)
        return _abs_unix_path(final_path)
    return ""


# --------------------------- API helpers ------------------------

def parse_category(category_url: str) -> Tuple[int, str]:
    """
    https://tiki.vn/sach-van-hoc/c839 -> (839, 'sach-van-hoc')
    """
    path = urlparse(category_url).path.strip("/")
    m = re.search(r"([^/]+)/c(\d+)", path)
    if not m:
        raise ValueError("URL danh mục không hợp lệ. Ví dụ: https://tiki.vn/sach-van-hoc/c839")
    return int(m.group(2)), m.group(1)


def listings_page(cid: int, urlkey: str, page: int, limit: int = 48) -> Dict:
    url = "https://tiki.vn/api/personalish/v1/blocks/listings"
    params = {
        "limit": limit,
        "aggregations": 2,
        "page": page,
        "category": cid,
        "urlKey": urlkey,
        "include": "advertisement",
    }
    return http_get_json(url, params=params)


def product_detail(pid: int) -> Dict:
    url = f"https://tiki.vn/api/v2/products/{pid}"
    return http_get_json(url)


def extract_author_and_desc(detail: Dict) -> Tuple[str, str]:
    authors = detail.get("authors") or []
    author_names = ", ".join([a.get("name", "") for a in authors if a.get("name")])
    desc = detail.get("short_description") or detail.get("description") or ""
    return author_names, clean_text(desc)


# ----------------------------- API MODE -------------------------

def crawl_api(category_url: str, out_csv: str, imgdir: str, max_pages: int, sleep: float, *, append: bool = False):
    cid, urlkey = parse_category(category_url)
    total = 0
    ensure_dir(imgdir)

    file_exists = os.path.exists(out_csv)
    mode = "a" if append else "w"
    with open(out_csv, mode, newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        if (not append) or (append and not file_exists):
            w.writerow(["image_path", "image_url", "title", "product_url", "author", "description"])

        for page in range(1, max_pages + 1):
            try:
                data = listings_page(cid, urlkey, page)
            except Exception as e:
                print(f"⚠️ Lỗi trang {page}: {e} → dừng.")
                break

            items = (data or {}).get("data") or []
            if not items:
                print(f"[*] Hết dữ liệu tại trang {page-1}.")
                break

            print(f"[*] Trang {page}: {len(items)} sp")
            for it in items:
                try:
                    pid = it.get("id")
                    title = clean(it.get("name", ""))
                    image_url = it.get("thumbnail_url") or ""
                    url_path = it.get("url_path") or ""
                    product_url = urljoin("https://tiki.vn", url_path) if url_path else ""

                    author, description = "", ""
                    if pid:
                        with suppress(Exception):
                            detail = product_detail(int(pid))
                            author, description = extract_author_and_desc(detail)

                    image_path = download_image(image_url, imgdir, title or f"book_{total+1}")
                    w.writerow([image_path, image_url, title, product_url, author, description])

                    total += 1
                    if total % 100 == 0:
                        print(f"  ✅ đã ghi {total} sp…")
                    if sleep > 0:
                        time.sleep(sleep)
                except Exception as e:
                    print("  ❌ lỗi item:", e)
                    continue
    print(f"✅ API xong: {total} sp | CSV: {out_csv} | Ảnh: {imgdir}/")


# ------------------------- BROWSER MODE -------------------------

def crawl_browser(category_url: str, out_csv: str, imgdir: str, max_steps: int, stable_needed: int, *, append: bool = False):
    from playwright.sync_api import sync_playwright

    seen_ids, items, lock = set(), [], threading.Lock()

    def on_response(resp):
        url = resp.url
        if "listings" not in url:
            return
        with suppress(Exception):
            data = resp.json()
            data_items = (data or {}).get("data") or []
            if not data_items:
                return
            with lock:
                for it in data_items:
                    pid = it.get("id")
                    if not pid or pid in seen_ids:
                        continue
                    title = clean(it.get("name", ""))
                    image_url = it.get("thumbnail_url") or ""
                    url_path = it.get("url_path") or ""
                    product_url = urljoin("https://tiki.vn", url_path) if url_path else ""
                    if not title or not image_url:
                        continue
                    seen_ids.add(pid)
                    items.append({
                        "id": int(pid),
                        "title": title,
                        "image_url": image_url,
                        "product_url": product_url,
                    })

    def scroll_smooth(page, steps=22, pause_ms=120):
        for _ in range(steps):
            page.evaluate("window.scrollBy(0, Math.floor(window.innerHeight*0.9));")
            page.wait_for_timeout(pause_ms)

    def click_all_see_more(page, max_clicks=6):
        clicks = 0
        while clicks < max_clicks:
            try:
                loc = page.locator("button:has-text('Xem thêm')")
                if loc.count() == 0 or not loc.first.is_visible():
                    loc = page.get_by_text("Xem thêm", exact=False)
                if loc.count() == 0 or not loc.first.is_visible():
                    break
                loc.first.click(timeout=2500)
                clicks += 1
                page.wait_for_timeout(900)
            except Exception:
                break
        return clicks

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(user_agent=HEADERS["User-Agent"])
        page.set_viewport_size({"width": 1280, "height": 1800})
        page.on("response", on_response)

        print("[*] Mở danh mục…")
        page.goto(category_url, timeout=180000)
        page.wait_for_load_state("domcontentloaded")
        page.wait_for_timeout(1200)

        print("[*] Cuộn + 'Xem thêm'…")
        stable, last_total = 0, 0
        for step in range(1, max_steps + 1):
            scroll_smooth(page)
            clicked = click_all_see_more(page)
            with suppress(Exception):
                page.wait_for_load_state("networkidle", timeout=3500)
            with lock:
                total_now = len(items)
            if step % 10 == 0:
                print(f"  • bước {step}: {total_now} sp (click {clicked})")
            if total_now == last_total:
                stable += 1
            else:
                stable, last_total = 0, total_now
            if stable >= stable_needed:
                break

        ensure_dir(imgdir)
        file_exists = os.path.exists(out_csv)
        mode = "a" if append else "w"
        with open(out_csv, mode, newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f, quoting=csv.QUOTE_ALL)
            if (not append) or (append and not file_exists):
                w.writerow(["image_path", "image_url", "title", "product_url", "author", "description"])
            with lock:
                rows = list(items)

            for i, it in enumerate(rows, 1):
                image_url, title = it["image_url"], it["title"]
                product_url = it.get("product_url", "")
                author, description = "", ""
                with suppress(Exception):
                    detail = product_detail(it["id"])
                    author, description = extract_author_and_desc(detail)

                image_path = download_image(image_url, imgdir, title or f"book_{i}")
                w.writerow([image_path, image_url, title, product_url, author, description])
                time.sleep(0.08)
        browser.close()
    print(f"✅ Browser xong: {len(items)} sp | CSV: {out_csv} | Ảnh: {imgdir}/")


# ------------------------- HỖ TRỢ NHIỀU URL --------------------

def derive_paths(cat_url: str, base_out: str, base_imgdir: str, multi: bool) -> Tuple[str, str]:
    """Tạo out/imgdir riêng theo urlkey + cid nếu có nhiều URL."""
    cid, urlkey = parse_category(cat_url)
    if not multi:
        return base_out, base_imgdir
    out = f"{urlkey}_{cid}.csv"
    imgdir = f"{urlkey}_{cid}"
    return out, imgdir


def expand_urls(category_urls: List[str] | None, category_file: str | None) -> List[str]:
    urls: List[str] = []
    if category_file:
        try:
            with open(category_file, "r", encoding="utf-8") as f:
                urls.extend([line.strip() for line in f if line.strip()])
        except Exception as e:
            print(f"⚠️ Không đọc được --category-file: {e}")
    if category_urls:
        urls.extend(category_urls)
    # loại trùng nhưng giữ thứ tự
    seen = set()
    uniq = []
    for u in urls:
        if u not in seen:
            uniq.append(u)
            seen.add(u)
    return uniq


# ------------------------------- MAIN ---------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Tiki covers crawler – API/Browser; hỗ trợ nhiều URL hoặc file danh sách.")

    ap.add_argument("--category-url", nargs="+", help="1 hoặc nhiều URL (cách nhau bởi khoảng trắng)")
    ap.add_argument("--category-file", help="Đường dẫn file chứa danh sách URL (mỗi dòng 1 URL)")

    ap.add_argument("--mode", choices=["api", "browser"], default="api",
                    help="api ổn định (rất nhiều) hoặc browser")
    ap.add_argument("--out", default="covers2.csv",
                    help="CSV đầu ra (mặc định: tách theo danh mục khi nhiều URL; dùng --merge để gộp 1 file)")
    ap.add_argument("--imgdir", default="covers2",
                    help="Thư mục lưu ảnh (mặc định: tách theo danh mục khi nhiều URL; dùng --merge để gộp 1 thư mục)")

    ap.add_argument("--merge", action="store_true",
                    help="Gộp mọi URL vào CHUNG 1 CSV (--out) và 1 thư mục ảnh (--imgdir)")

    # API
    ap.add_argument("--max-pages", type=int, default=5000, help="API: số trang listings tối đa")
    ap.add_argument("--sleep", type=float, default=0.15, help="API: delay nhẹ mỗi item")

    # Browser
    ap.add_argument("--max-steps", type=int, default=1500, help="Browser: vòng cuộn tối đa")
    ap.add_argument("--stable-needed", type=int, default=40, help="Browser: số vòng không tăng trước khi dừng")

    args = ap.parse_args()

    urls = expand_urls(args.category_url, args.category_file)
    if not urls:
        ap.error("Cần --category-url ... hoặc --category-file ...")

    multi = (len(urls) > 1) and (not args.merge)

    first = True
    for u in urls:
        out, imgdir = (args.out, args.imgdir) if args.merge else derive_paths(u, args.out, args.imgdir, multi)
        try:
            if args.mode == "api":
                crawl_api(u, out, imgdir, args.max_pages, args.sleep, append=(args.merge and not first))
            else:
                crawl_browser(u, out, imgdir, args.max_steps, args.stable_needed, append=(args.merge and not first))
        except KeyboardInterrupt:
            print("Đã hủy bởi người dùng.")
            sys.exit(130)
        except Exception as e:
            print(f"❌ Bỏ qua {u}: {e}")
        finally:
            first = False


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nĐã hủy bởi người dùng.")
        sys.exit(130)
