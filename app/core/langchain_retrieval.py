from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

from models.chat_history_model import ChatHistory
from .base import chat_llm, chroma_vector_store

from py2neo import Graph

# 初始化 Neo4j 图数据库连接
graph = Graph("bolt://localhost:7687", auth=("neo4j", "lrq20041224"))

from zhipuai import ZhipuAI

# 初始化 ZhipuAI 客户端
client = ZhipuAI(api_key="b05241d40957460c8b15d23f0cbabfda.maAt6j0cQNNa8GPD")


def build_history_template(chat_history_list: list[ChatHistory]):
    """构建聊天历史模板"""
    if not isinstance(chat_history_list, list) or len(chat_history_list) == 0:
        return []
    history_messages: list[BaseMessage] = []
    for history in chat_history_list:
        if history.role == "user":
            history_messages.append(HumanMessage(content=history.content))
        elif history.role == "assistant":
            history_messages.append(AIMessage(content=history.content))
    return history_messages


def extract_entity_llm(question: str) -> str:
    prompt = f"""从下面的问题中提取用户查询的核心电影或剧集关键词（可以是电影名、演员名、导演名等）：
问题：{question}
输出格式：只返回关键词，如：7号房的礼物"""
    response = client.chat.completions.create(
        model="GLM-4-Flash-250414",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    return response.choices[0].message.content.strip()


def extract_search_query_from_question(question: str) -> str:
    prompt = f"""请将下面的问题转化为用于搜索文档的关键词组合，去掉冗余表达，保留核心信息：
问题：{question}
只输出关键词组合，如：“评分高 感人 英文电影”
"""
    try:
        response = client.chat.completions.create(
            model="GLM-4-Flash-250414",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"关键词抽取失败：{e}")
        return question  # 回退为原始问题


def kg_query(question: str) -> str:
    keyword = extract_entity_llm(question)
    if not keyword:
        print("未提取到有效关键词")
        return ""

    cypher = f"""
    MATCH (s)-[r]->(o)
    WHERE s.name CONTAINS '{keyword}' OR o.name CONTAINS '{keyword}'
    RETURN s.name AS subject, type(r) AS predicate, o.name AS object
    LIMIT 10
    """

    try:
        print("\n[KG 查询语句]")
        print(cypher.strip())

        results = graph.run(cypher).data()

        print("\n[KG 查询结果]")
        if not results:
            print("未找到相关三元组")
            return ""
        triples = [
            f"({r['subject']}，{r['predicate']}，{r['object']})" for r in results
        ]
        return "以下是与问题相关的知识图谱三元组：\n" + "\n".join(triples)
    except Exception as e:
        return f"[KG查询失败：{str(e)}]"


def combine_kg_and_docs(question: str, retriever) -> str:
    """融合文档检索与知识图谱信息作为上下文"""
    query = extract_search_query_from_question(question)
    docs = retriever.invoke(query)
    doc_text = (
        "\n".join(
            [f"【文档片段{i+1}】\n{doc.page_content}" for i, doc in enumerate(docs)]
        )
        if docs
        else ""
    )

    kg_triples = kg_query(question)
    if kg_triples:
        kg_text = f"""【知识图谱查询过程】
我在知识图谱中搜索了与“{question}”相关的实体和关系，找到以下三元组：
{kg_triples}
"""
    else:
        kg_text = ""

    if kg_text and doc_text:
        return f"{kg_text}\n【补充文档信息】\n{doc_text}"
    elif kg_text:
        return kg_text
    else:
        return doc_text or "未找到相关知识图谱或文档信息。"


from langchain_core.tools import Tool
from duckduckgo_search import DDGS
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType


# 定义搜索工具函数
# ----------------- 搜索工具 -----------------
def web_search(query: str, max_results: int = 5):
    try:
        with DDGS() as ddgs:
            return [
                f"{i+1}. {r.get('title','')} - {r.get('url','')}\n    {r.get('body','')}"
                for i, r in enumerate(ddgs.text(query, max_results=max_results))
            ]
    except Exception:
        return []


search_tool = Tool(
    name="web_search", func=web_search, description="联网实时搜索电影/剧集信息"
)
from langchain.memory import ConversationBufferMemory

# 初始化记忆组件
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


def build_qa_chain():
    """构建融合了文档 + 知识图谱的问答链"""
    vector_store = chroma_vector_store()

    # ② LLM & 记忆
    llm = chat_llm()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # 初始化检索，并配置
    # 使用mmr的检索算法，
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 3,
            "fetch_k": 20,
            "lambda_mult": 0.5,
        },
    )

    # 提示词模板
    system_template = """
    你是一位专业的电影和剧集信息问答助手，具备强大的知识检索与理解能力，能够帮助用户准确、高效地查询电影与电视剧的各种信息。

    你可以基于提供的文档或检索内容，回答以下类型的问题：
    - 查询某部电影或剧集的演职员信息（导演、演员、编剧等）
    - 概括剧情简介或核心看点
    - 提供观众评分与评论摘要
    - 推荐与某部作品相似的其他作品
    - 根据类型（如科幻、爱情、悬疑等）或上映年代筛选作品

    【特别要求】：
    - 你应体现**你通过查询知识图谱三元组得到的推理过程**
    - 若知识图谱中包含相关三元组，应明确展示出来，例如：“根据知识图谱中的三元组 (柳承龙，主演，7号房的礼物)...”
    - 同时需要使用文档检索内容，但也请说明信息来源

    你不可凭空编造。当知识图谱与文档均无相关内容时，你应联网搜索相关的电影信息并回复，
    检索结果：  
    【联网搜索】\n{web_results}\n\n

    如果联网没有搜索到相关的电影信息，则回复：“抱歉，暂时没有相关信息。”

    以下是知识图谱与文档内容：
    {context}
    """

    prompt = ChatPromptTemplate(
        [
            ("system", system_template),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}"),
        ]
    )

    return (
        {
            "context": lambda x: combine_kg_and_docs(x["question"], retriever),
            "web_results": lambda x: search_tool.func(x["question"]),
            "chat_history": lambda x: x["chat_history"],
            "question": lambda x: x["question"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )
