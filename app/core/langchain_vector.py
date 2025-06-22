import unicodedata
from fastapi import HTTPException
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
import time

from .base import LOAD_PATH, VECTOR_DIR, chroma_vector_store


def clean_text(text: str) -> str:
    """统一文本清洗函数"""

    cleaned = ""
    if not text.strip():
        return cleaned
    # 1. 标准化全角字符（字母、数字、标点）为半角
    normalized = unicodedata.normalize("NFKC", text)
    # 2. 删除所有空格（包括全角空格\u3000和普通空格）
    cleaned = normalized.replace("\u3000", "").replace(" ", "")
    # 3. 中文标点替换为英文标点（按需扩展）
    replacements = {
        "，": ",",
        "。": ".",
        "（": "(",
        "）": ")",
        "；": ";",
        "：": ":",
        "！": "!",
        "？": "?",
    }
    for cn, en in replacements.items():
        cleaned = cleaned.replace(cn, en)
    return cleaned


def load_documents(source_dir=LOAD_PATH):
    """
    加载指定目录下的所有文档
    支持格式：.txt, .pdf, .docx, .md
    """

    try:
        # 分别加载不同格式
        text_loader = DirectoryLoader(
            path=source_dir,  # 指定读取文件的父目录
            glob=["**/*.txt", "**/*.md"],  # 指定读取文件的格式
            show_progress=True,  # 显示加载进度
            use_multithreading=True,  # 使用多线程
            loader_cls=TextLoader,  # 指定加载器
            loader_kwargs={"encoding": "utf-8"},  # 自动检测文件编码,
        )

        pdf_loader = DirectoryLoader(
            path=source_dir,
            glob="**/*.pdf",
            show_progress=True,
            use_multithreading=True,
            loader_cls=PyPDFLoader,
        )

        docx_loader = DirectoryLoader(
            path=source_dir,
            glob="**/*.docx",
            show_progress=True,
            use_multithreading=True,
            loader_cls=Docx2txtLoader,
            loader_kwargs={"autodetect_encoding": True},
        )

        # 初步清洗 PDF 文档的文本，删除多余空格。
        # TODO: 后续会修改，将单独优化 PDF 文档的分割。
        pdf_docs = pdf_loader.load()
        for doc in pdf_docs:
            doc.page_content = clean_text(doc.page_content)

        # 合并文档列表
        docs = []
        docs.extend(text_loader.load())
        docs.extend(pdf_docs)
        docs.extend(docx_loader.load())
        print(f"成功加载 {len(docs)} 份文档")
        return docs
    except Exception as e:
        print(f"加载文档失败：{str(e)}")
        raise HTTPException(status_code=500, detail=f"加载文档失败：{str(e)}")


def split_documents(documents, chunk_size=800, chunk_overlap=150):
    """
    使用递归字符分割器处理文本
    参数说明：
    - chunk_size：每个文本块的最大字符数，推荐 500-1000
    - chunk_overlap：相邻块之间的重叠字符数（保持上下文连贯），推荐 100-200
    """
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", "。", "!", "?", "？", "！", "；", ";"],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,  # 保留原始文档中的位置信息
    )

    split_docs = text_splitter.split_documents(documents)
    print(f"原始文档数：{len(documents)}")
    print(f"分割后文本块数：{len(split_docs)}")

    return split_docs


def create_vector_store(split_docs, persist_dir=VECTOR_DIR):
    """
    创建持久化向量数据库
    - split_docs: 经过分割的文档列表
    - persist_dir: 向量数据库存储路径（建议使用WSL原生路径）
    """

    # 初始化 Chroma 向量数据库
    vector_store = chroma_vector_store()

    # 向量化文档之前，先把原来集合里的数据清空
    ids = vector_store._collection.get()["ids"]
    if len(ids):
        vector_store.delete(ids=vector_store._collection.get()["ids"])

    # 如果分割文档为空，不做向量化操作
    if not split_docs or len(split_docs) == 0:
        return

    try:
        start_time = time.time()
        print(f"\n开始向量化====>")

        # 向量化文档到向量数据库
        vector_store.add_documents(split_docs)

        print(f"\n向量化完成！耗时 {time.time()-start_time:.2f} 秒")
        print(f"数据库存储路径：{persist_dir}")
        print(f"总文档块数：{vector_store._collection.count()}")

    except Exception as e:
        print(f"向量化失败：{str(e)}")
        raise HTTPException(status_code=500, detail=f"向量化失败：{str(e)}")


def vector_documents():
    """
    启动文档向量化，并保存数据库
    """
    # 加载本地文档
    documents = load_documents()
    # 执行分割
    split_docs = split_documents(documents)

    # 执行向量化（使用之前分割好的split_docs）
    create_vector_store(split_docs)
    # 抽取三元组并存入KG
    triplets = process_documents(documents)
    # store_triplets_to_neo4j(triplets)


def triplet_extraction_prompt(text: str) -> str:
    return f"""请从以下文本中提取三元组（实体1，关系，实体2）：
文本：{text}
输出格式：[(实体1, 关系, 实体2), ...]
"""


from openai import OpenAI  # or use any other LLM API you prefer

from langchain_community.chat_models import ChatOpenAI  # DeepSeek 接口
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import ast  # 更安全的字符串转结构方式

API_KEY = "sk-h7jNnP-gm7tAY-c3ox8nSw"
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

MODEL_NAME = "deepseek-r1:7b"


import re
import os
from langchain.chat_models import ChatOpenAI

# 设置环境变量
os.environ["OPENAI_API_KEY"] = API_KEY
os.environ["OPENAI_API_BASE"] = "https://llmapi.blsc.cn/v1"  # 自定义模型地址
from zhipuai import ZhipuAI

# 初始化 ZhipuAI 客户端
client = ZhipuAI(api_key="b05241d40957460c8b15d23f0cbabfda.maAt6j0cQNNa8GPD")


import re


def extract_triplets_from_text(text: str) -> list[tuple]:
    prompt = f"""请从以下电影介绍中提取有价值的知识三元组（实体1，关系，实体2），要求：
1. 仅提取对电影信息有帮助的事实性关系，如：导演、主演、编剧、上映时间、上映地区、改编自、类型、评分、语言、出品方、角色关系（如A 饰演 B）；
2. 关系必须是有意义的事实，不要提取主观感受或模糊情绪，如“感人”“好看”“戳人”等；
3. 如果没有可提取的三元组，请返回空列表 [];
4. 输出格式严格为 Python 列表格式，如：[(实体1, 关系, 实体2), ...]；
5. 输出中实体和关系都用半角双引号括起来，如 ("柳承龙", "主演", "7号房的礼物")

文本内容：
{text}
"""

    try:
        response = client.chat.completions.create(
            model="GLM-4-Flash-250414",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )

        result_str = response.choices[0].message.content.strip()
        print(f"ZhipuAI返回: {result_str}")

        # 使用正则提取形如 ("实体1", "关系", "实体2") 的三元组
        matches = re.findall(r'\("([^"]+)",\s*"([^"]+)",\s*"([^"]+)"\)', result_str)

        triplets = [(e1.strip(), rel.strip(), e2.strip()) for e1, rel, e2 in matches]
        return triplets

    except Exception as e:
        print(f"提取三元组失败: {e}")
        return []


from py2neo import Graph, Node, Relationship
from neo4j import GraphDatabase

# graph = Graph("bolt://localhost:7687", auth=("neo4j", "lrq20041224"))
neo4j_driver = GraphDatabase.driver(
    "bolt://localhost:7687",  # 你的 Neo4j 地址
    auth=("neo4j", "lrq20041224"),  # 你的用户名和密码
)


# 插入三元组到Neo4j
def insert_triplet_to_neo4j(triplet):
    with neo4j_driver.session() as session:
        e1, rel, e2 = triplet
        session.run(
            """
            MERGE (h:Entity {name: $e1})
            MERGE (t:Entity {name: $e2})
            MERGE (h)-[:RELATION {type: $rel}]->(t)
            """,
            e1=e1,
            rel=rel,
            e2=e2,
        )


def process_documents(documents):
    all_triplets = []
    for doc in documents:
        content = doc.page_content[:2000]  # 可酌情截断
        triplets = extract_triplets_from_text(content)

        for triplet in triplets:
            print(f"插入三元组: {triplet}")
            insert_triplet_to_neo4j(triplet)

        all_triplets.extend(triplets)

    print(f"共提取并插入三元组数: {len(all_triplets)}")
    return all_triplets


# def vector_documents_with_kg():
#     documents = load_documents()
#     split_docs = split_documents(documents)

#     # 向量化原始内容
#     create_vector_store(split_docs)

#     # ⬇️ 新增：抽取三元组并存入KG
#     triplets = process_documents(split_docs)
#     # store_triplets_to_neo4j(triplets)
