from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Dùng SQLite, file database sẽ tên là booksearch.db
SQLALCHEMY_DATABASE_URL = "sqlite:///./booksearch.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Hàm helper để lấy DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()