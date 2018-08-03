from sqlalchemy.ext.declarative import declarative_base
import datetime 
from sqlalchemy import Column, Integer, String, create_engine ,DateTime

# 宣告對映
Base = declarative_base()


class File(Base):
    __tablename__ = 'box'
    id = Column(String(50), primary_key=True)
    root = Column(String(100), nullable=False)
    CreateTime = Column(DateTime(timezone=True))
    Value = Column(String(250),nullable=False)


# 連結SQLite3資料庫example.db
engine = create_engine('sqlite:///example.db')

# 建立Schema
Base.metadata.create_all(engine)    # 相當於Create Table