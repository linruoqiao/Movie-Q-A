from typing import Annotated
import uuid
from fastapi import APIRouter, Form, Query
from fastapi.responses import FileResponse
from core.langchain_vector import vector_documents
from crud.document_crud import DocumentCrud
from models.document_model import (
    DocumentParams,
    DocumentResponse,
    UpdateFormData,
    UploadFormData,
    URLRequest
)
import datetime
from urlToTxt import URLTextExtractor
from urllib.parse import quote
from routers.base import success,failure


router = APIRouter(
    prefix="/documents",
    tags=["documents"],
    responses={404: {"message": "您所访问的资源不存在！"}},
)

document_crud = DocumentCrud()


@router.post("/add")
async def add_doc(data: Annotated[UploadFormData, Form()]):
    await document_crud.add(data)
    return success(None, "添加成功！")


@router.put("/update")
async def update_doc(data: Annotated[UpdateFormData, Form()]):
    await document_crud.update(data)
    return success(None, "更新成功！")


@router.get("/page", response_model=DocumentResponse)
async def page_doc(params: Annotated[DocumentParams, Query()]):
    result = document_crud.page(params)
    return success(result)


@router.delete("/delete")
async def del_doc(data: DocumentParams):
    await document_crud.delete(data.id)
    return success(None, "删除成功！")


@router.get("/read/{item_id}")
async def read_doc_file(item_id: uuid.UUID):
    file_path, real_name = document_crud.download(item_id)
    header_file_name = quote(real_name, encoding="utf-8")
    headers = {"Content-Disposition": f"inline; filename*=UTF-8''{header_file_name}"}
    return FileResponse(path=file_path, headers=headers, media_type=None)


@router.get("/vector-all")
async def vector_docs():
    vector_documents()
    document_crud.vector_all_docs()
    return success(None, "已全部向量化。")


@router.post("/url_to_text")
async def get_url_text(request_data: URLRequest):
    extractor = URLTextExtractor()
    print("✅ 收到 URL 提交请求")
    try:
        doc_name = request_data.name
        url = request_data.url

        result = extractor.extract_from_url(url)['text']
        print(result)

        # 保存到数据库
        db_document = await document_crud.add_from_url(doc_name, url, result)

        return success({
            "document_id": db_document.id,
            "document_name": db_document.name,
            "document_url": url,
            "content": result,
            "status": "processed"
        }, "网页内容提取并保存成功！")

    except Exception as e:
        return failure(500,f"处理URL文档失败: {str(e)}")

