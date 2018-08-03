from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import datetime
from db import Base, File

# 連結到之前建立的資料庫
engine = create_engine('sqlite:///example.db')
Base.metadata.bind = engine

# 建立Session
DBSession = sessionmaker(bind=engine)
# 或是分成二段的寫法。這是用在暫時尚無設定engine，待有engine時在第二段連結到engine
# DBSession = sessionmaker()
# DBSession.bind = engine  # 或是 DBSession.configure(bind=engine)
session = DBSession()

# 建立一對映類別的實例
new_user = File(id='test' , Value ='11', root = '//ss')

# 新增new_user
# session.add(new_user)
# session.commit()    # 寫入。在commit()之前，也可rollback()

# 查詢
users = session.query(File).filter(File.CreateTime >= datetime.date.today()).all()
print(datetime.date.today())
for File in users:
    print (File.CreateTime)
# 還有其他種查詢的方法：
# session.query(User).all()
# session.query(User).filter(User.id == 1).one()