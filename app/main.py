import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from crud.base import create_db_and_tables
import imagesolver
from routers.base import failure

# 导入 fastApi 子模块
from routers import chat_router
from routers import chat_session_router
from routers import document_router

# 上传文件
import os
import shutil
import tempfile
from typing import List

from fastapi import File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from flask_cors import CORS

# FastAPI 主入口
app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源（生产环境应限制）
    allow_methods=["*"],  # 允许所有方法
)

# 将 fastApi 子模块整合到 app 中
app.include_router(chat_router.router)
app.include_router(chat_session_router.router)
app.include_router(document_router.router)


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request, exc):
    """
    重写 fastApi 错误信息
    """
    return JSONResponse(
        failure(exc.status_code, exc.detail), status_code=exc.status_code
    )


@app.get("/")
def read_root():
    return {"code": 200, "message": "已启动服务。"}


ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.post("/analysis_image", response_model=dict)
async def analysis_image(
    files: List[UploadFile] = File(..., description="一张或多张待分析的图片")
):
    if not files or len(files) == 0:
        raise HTTPException(status_code=400, detail="未找到图片")

    # 创建一个临时目录保存图片 图片数量不超过10张
    with tempfile.TemporaryDirectory() as tmp_dir:
        image_paths = []

        for file in files:
            if not allowed_file(file.filename):
                raise HTTPException(
                    status_code=400, detail="仅支持 png/jpg/jpeg 格式图片"
                )

            # 保存文件到临时目录
            temp_path = os.path.join(tmp_dir, file.filename)
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            image_paths.append(temp_path)

        try:
            # 调用多图分析函数，传入路径列表
            prompts = imagesolver.analyze_multiple_images(image_paths)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"服务异常: {str(e)}")

    return JSONResponse(
        {"code": 200, "message": "分析完成", "result": prompts}, status_code=200
    )


if __name__ == "__main__":
    # 创建或启动数据库
    create_db_and_tables()
    # 启动 uvicorn 服务
    uvicorn.run("main:app", port=8082, log_level="info", reload=True)
