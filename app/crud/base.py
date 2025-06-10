from typing import Annotated
from fastapi import Depends
from sqlmodel import SQLModel, Session, create_engine

# Session：数据库会话对象，用来执行增删改查
# create_engine：创建数据库连接引擎

# 导入所有数据库表
# document_model：定义了“文档”表的结构，比如文档ID、内容、标题等
# chat_session_model：每次问答会话的记录
# chat_history_model：历史聊天记录

from models import document_model, chat_session_model, chat_history_model

# 创建数据库
sqlite_file_name = "document_qa.db"
# 在项目根目录创建数据库文件
sqlite_url = f"sqlite:///{sqlite_file_name}"
# echo：打印所有的SQL执行语句
engine = create_engine(sqlite_url, echo=True)


# 会自动读取通过SQLModel定义的所有数据表模型，并在数据库中创建表结构
def create_db_and_tables():
    """创建数据库和所有表"""
    SQLModel.metadata.create_all(engine)


# Session（engine）表示和数据库建立连接
# yield session 让 FastAPI 每次请求时都能用这个数据库会话进行操作（比如查数据、写数据）
def get_session():
    """依赖注入，将engine注册到FastApi session中"""
    with Session(engine) as session:
        yield session


# 当某个接口依赖 SessionDep 类型时，就自动注入 get_session() 提供的数据库会话；
SessionDep = Annotated[Session, Depends(get_session)]
