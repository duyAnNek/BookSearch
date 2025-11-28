from pydantic import BaseModel
from typing import Optional, List

# Model cho dữ liệu trả về (an toàn, không lộ cấu trúc DB)
class BookBase(BaseModel):
    id: str # (normalized_path)
    image_path: str
    title: Optional[str] = None
    author: Optional[str] = None
    
    class Config:
        orm_mode = True

class BookDetail(BookBase):
    description: Optional[str] = None
    product_url: Optional[str] = None
    image_url: Optional[str] = None
    
    class Config:
        orm_mode = True

# Dùng cho kết quả tìm kiếm (chỉ cần thông tin cơ bản)
class BookSearchResult(BookBase):
    pass

class ViltSearchResult(BaseModel):
    index: int
    score: float
    image: Optional[str] = None
    text: Optional[str] = None