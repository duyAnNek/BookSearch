import os
from typing import List
from google import genai

# Dùng tên model hợp lệ cho google-genai v1beta
GEMINI_MODEL = "gemini-2.5-flash"

class GeminiSearchSuggestor:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set")
        self.client = genai.Client(api_key=api_key)

    def suggest_queries(self, query: str, max_suggestions: int = 5) -> List[str]:
        prompt = f"""
        Bạn là trợ lý tìm kiếm sách tiếng Việt.
        Người dùng đang nhập truy vấn: "{query}".

        Hãy gợi ý tối đa {max_suggestions} câu truy vấn tìm sách,
        ngắn gọn, tự nhiên, tiếng Việt, không giải thích thêm,
        mỗi gợi ý một dòng.

        Chỉ trả về danh sách câu truy vấn, không đánh số.
        """
        resp = self.client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )

        text = resp.text or ""
        lines = [l.strip("-• ").strip() for l in text.splitlines()]
        lines = [l for l in lines if l]
        return lines[:max_suggestions]