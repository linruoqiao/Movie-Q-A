import os
from pathlib import Path
import uuid
import aiofiles
import aiofiles.os
from fastapi import File, HTTPException, UploadFile
from sqlmodel import Session, desc, func, select
from models.document_model import (
    Document,
    DocumentFormat,
    DocumentParams,
    DocumentUpdate,
    UpdateFormData,
    UploadFormData,
)
from .base import engine
from uuid import uuid4

# 文档管理系统，包括：上传（add）、修改（update）、删除（delete）
# 分页查询（page）、下载（download）、批量打标向量状态（vector_all_docs）


class DocumentCrud:
    __BASE_PATH = "E://code//LLM_RAG//Movie-Q-A//app"
    __FILE_PATH = "/fileStorage"
    async def __save_file(self, file: UploadFile = File()):
        """I/O操作，写入文件"""

        old_file_name = os.path.basename(file.filename)
        # 生成唯一文件名称
        new_file_name = f"{uuid.uuid4()}{Path(old_file_name).suffix}"
        save_path = Path(self.__BASE_PATH + self.__FILE_PATH) / new_file_name
        new_path = Path(self.__FILE_PATH) / new_file_name
        new_file_path = str(new_path)
        new_suffix = new_path.suffix

        # 利用 aiofiles 保存文件到指定目录
        async with aiofiles.open(save_path, "wb") as buffer:
            while content := await file.read(1024000):  # 每次读取 1KB
                await buffer.write(content)
        return new_file_name, new_file_path, new_suffix

    async def __delete_file(self, path: str):
        """I/O操作，删除文件"""
        if await aiofiles.os.path.exists(path):
            await aiofiles.os.remove(path)

    async def add(self, data: UploadFormData):
        """新增文档记录"""
        file = data.file
        new_file_name, new_file_path, new_suffix = await self.__save_file(file)
        doc = DocumentUpdate(
            name=data.name,
            file_name=new_file_name,
            file_path=new_file_path,
            suffix=new_suffix,
            vector="",
        )

        with Session(engine) as session:
            db_document = Document.model_validate(doc)
            session.add(db_document)
            session.commit()
            session.refresh(db_document)
            return db_document

    async def update(self, data: UpdateFormData):
        """修改文档记录"""
        file = data.file
        doc = DocumentUpdate(
            name=data.name,
            file_name=data.file_name,
            file_path=data.file_path,
            vector=data.vector,
        )

        with Session(engine) as session:
            db_document = session.get(Document, data.id)
            if not db_document:
                raise HTTPException(status_code=404, detail="找不到该记录。")

            if file:
                # 数据库查找旧的文件名，并删除文件
                path = Path(self.__BASE_PATH + db_document.file_path)
                await self.__delete_file(path)
                # 保存新的文件
                new_file_name, new_file_path, new_suffix = await self.__save_file(file)
                doc.file_name = new_file_name
                doc.file_path = new_file_path
                doc.suffix = new_suffix
                doc.vector = ""

            doc = doc.model_dump(exclude_unset=True)
            db_document.sqlmodel_update(doc)
            session.add(db_document)
            session.commit()
            session.refresh(db_document)
            return db_document

    async def delete(self, id: str):
        """删除文档记录，并删除文件"""
        with Session(engine) as session:
            document = session.get(Document, id)
            if not document:
                raise HTTPException(status_code=404, detail="文档未找到")

            path = Path(self.__BASE_PATH + document.file_path)
            await self.__delete_file(path)

            session.delete(document)
            session.commit()

    def page(self, params: DocumentParams):
        """查询文档记录，分页"""
        offset = params.page_num * params.page_size
        limit = params.page_size
        name = params.name
        with Session(engine) as session:
            count_query = select(func.count(Document.id))
            query = select(Document)

            # 根据 name 模糊查询，并查询总数
            if name:
                count_query = count_query.where(Document.name.contains(name))
                query = query.where(Document.name.contains(name))

            # 没有进行赋值？
            query.order_by(desc(Document.date)).offset(offset).limit(limit)

            total = session.exec(count_query).one()
            items = session.exec(query).all()
            list = [DocumentFormat.model_validate(d) for d in items]
            return {
                "total": total,
                "page_num": params.page_num,
                "page_size": params.page_size,
                "list": list,
            }

    # 下载文件，根据uuid获取文档记录
    def download(self, item_id: uuid.UUID):
        with Session(engine) as session:
            document = session.get(Document, item_id)
            if not document:
                raise HTTPException(status_code=404, detail="文档未找到")

            file_path = Path(self.__BASE_PATH + document.file_path)
            real_name = document.name + file_path.suffix

            return file_path, real_name

    # 将数据库中所有文档的vector字段改为“yes”
    def vector_all_docs(self):
        with Session(engine) as session:
            query = select(Document)
            doc_list = session.exec(query).all()
            for doc in doc_list:
                doc.vector = "yes"
                session.add(doc)
            session.commit()

    async def add_from_url(self, name: str, url: str, content: str):
        """通过URL创建文档记录（不上传文件）"""
        # 使用 UUID 生成一个虚拟文件名，模拟文件信息
        new_file_name = f"{uuid4()}.txt"
        save_path = Path(self.__BASE_PATH + self.__FILE_PATH) / new_file_name
        new_path = Path(self.__FILE_PATH) / new_file_name
        new_file_path = str(new_path)
        suffix = "txt"
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(content)
        # 创建 DocumentUpdate 对象（或 Document 对象）
        doc = DocumentUpdate(
            name=name,
            file_name=new_file_name,
            file_path=new_file_path,
            suffix=suffix,
            vector="",
        )

        with Session(engine) as session:
            db_document = Document.model_validate(doc)
            session.add(db_document)
            session.commit()
            session.refresh(db_document)

            # 可选：另建一张表存储正文内容，也可以在Document中加一个 content 字段
            return db_document