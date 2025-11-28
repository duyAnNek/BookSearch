from sqlalchemy import Column, Integer, String, Text
from .database import Base

class Book(Base):
    __tablename__ = "books"

    # Chúng ta dùng 'image_path' chuẩn hóa làm ID chính
    # ví dụ: 'all_covers/img1.jpg'
    id = Column(String, primary_key=True, index=True) 
    
    image_path = Column(String, unique=True, index=True)
    image_url = Column(String, nullable=True)
    title = Column(String, index=True)
    product_url = Column(String, nullable=True)
    author = Column(String, nullable=True, index=True)
    description = Column(Text, nullable=True)