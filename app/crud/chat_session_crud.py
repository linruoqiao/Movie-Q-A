from fastapi import HTTPException
from sqlmodel import Session, desc, select
from models.chat_session_model import ChatSession, ChatSessionParams, ChatSessionUpdate

from .base import engine


# 会话记录的 CRUD 操作类
class ChatSessionCrud:
    # 保存（新增或更新）会话记录
    def save(self, data: ChatSessionParams):
        # 创建用于更新的数据对象
        chat_session = ChatSessionUpdate(title=data.title)
        # 如果 data 中包含 id，说明是更新已有会话
        if data.id:
            with Session(engine) as session:
                db_update_session = session.get(ChatSession, data.id)
                chat_session = chat_session.model_dump(exclude_unset=True)
                db_update_session.sqlmodel_update(chat_session)
                session.add(db_update_session)
                session.commit()
                session.refresh(db_update_session)
                return db_update_session
        # 新增会话
        with Session(engine) as session:
            db_add_session = ChatSession.model_validate(chat_session)
            session.add(db_add_session)
            session.commit()
            session.refresh(db_add_session)
            return db_add_session

    # 获取所有会话记录的列表，按时间倒序排序
    def list(self):
        with Session(engine) as session:
            query = select(ChatSession).order_by(desc(ChatSession.date))
            chat_session_list = session.exec(query).all()
            return chat_session_list

    def delete(self, id: str):
        """删除会话记录"""
        with Session(engine) as session:
            db = session.get(ChatSession, id)
            if not db:
                raise HTTPException(status_code=500, detail="会话记录不存在。")

            session.delete(db)
            session.commit()
