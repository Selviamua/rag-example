"""批量问答脚本 - 读取 benchmark.json 中的问题并生成答案"""

# Copyright (c) 2023 Brent Benson
#
# This file is part of [project-name], licensed under the MIT License.
# See the LICENSE file in this repository for details.

import os
import json
import pprint
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

# Load the environment variables from the .env file
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import AzureChatOpenAI, BedrockChat
from langchain_community.vectorstores import Chroma, OpenSearchVectorSearch
from langchain_community.vectorstores.pgvector import PGVector
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferWindowMemory

# Log full text sent to LLM
VERBOSE = False

# Details of persisted embedding store index
COLLECTION_NAME = "doc_index"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Size of window for buffered window memory
MEMORY_WINDOW_SIZE = 10

# 输入输出文件路径
BENCHMARK_FILE = "benchmark.json"
OUTPUT_FILE = f"qa_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"


def load_llm():
    """加载语言模型"""
    # DeepSeek 配置（从 .env 读取）
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    deepseek_model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    deepseek_base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    
    openai_model_name = os.getenv("OPENAI_MODEL_NAME")
    aws_credential_profile_name = os.getenv("AWS_CREDENTIAL_PROFILE_NAME")
    aws_bedrock_model_name = os.getenv("AWS_BEDROCK_MODEL_NAME")
    
    if deepseek_api_key:
        print("Using DeepSeek for language model.")
        return ChatOpenAI(
            model=deepseek_model,
            api_key=deepseek_api_key,
            base_url=deepseek_base_url,
            temperature=0.5,
            verbose=VERBOSE,
        )
    elif openai_model_name:
        print("Using Azure for language model.")
        return AzureChatOpenAI(
            temperature=0.5, deployment_name=openai_model_name, verbose=VERBOSE
        )
    elif aws_credential_profile_name and aws_bedrock_model_name:
        print("Using Amazon Bedrock for language model.")
        return BedrockChat(
            credentials_profile_name=aws_credential_profile_name,
            model_id=aws_bedrock_model_name,
            verbose=VERBOSE,
        )
    else:
        raise EnvironmentError("No language model environment variables found.")


def get_embed_db(embeddings):
    """获取向量数据库"""
    chroma_persist_dir = os.getenv("CHROMA_PERSIST_DIR")
    opensearch_url = os.getenv("OPENSEARCH_URL")
    postgres_conn = os.getenv("POSTGRES_CONNECTION")
    
    if chroma_persist_dir:
        db = get_chroma_db(embeddings, chroma_persist_dir)
    elif opensearch_url:
        db = get_opensearch_db(embeddings, opensearch_url)
    elif postgres_conn:
        db = get_postgres_db(embeddings, postgres_conn)
    else:
        raise EnvironmentError("No vector store environment variables found.")
    return db


def get_chroma_db(embeddings, persist_dir):
    """获取 Chroma 向量数据库"""
    db = Chroma(
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=persist_dir,
    )
    return db


def get_opensearch_db(embeddings, url):
    """获取 OpenSearch 向量数据库"""
    username = os.getenv("OPENSEARCH_USERNAME")
    password = os.getenv("OPENSEARCH_PASSWORD")
    db = OpenSearchVectorSearch(
        embedding_function=embeddings,
        index_name=COLLECTION_NAME,
        opensearch_url=url,
        http_auth=(username, password),
        use_ssl=False,
        verify_certs=False,
        ssl_assert_hostname=False,
        ssl_show_warn=False,
    )
    return db


def get_postgres_db(embeddings, connection_string):
    """获取 PostgreSQL 向量数据库"""
    db = PGVector(
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
        connection_string=connection_string,
    )
    return db


def load_benchmark_questions(benchmark_file):
    """加载 benchmark 问题"""
    try:
        with open(benchmark_file, 'r', encoding='utf-8') as f:
            questions = json.load(f)
        print(f"成功加载 {len(questions)} 个问题")
        return questions
    except FileNotFoundError:
        print(f"错误: 找不到文件 {benchmark_file}")
        return []
    except json.JSONDecodeError as e:
        print(f"错误: JSON 解析失败 - {e}")
        return []


def evaluate_answer_correctness(answer, gold_answer):
    """评估答案是否正确
    使用语义相似度而不是严格字符串匹配
    """
    answer = answer.strip()
    gold_answer = gold_answer.strip()
    
    # 如果是完全相等，那肯定是对的
    if answer == gold_answer:
        return True
    
    # 检查金标准答案是否包含在生成的答案中（关键内容）
    if gold_answer in answer:
        return True
    
    # 对于数值答案，提取并比较（如 "730 mg/L"）
    import re
    gold_numbers = re.findall(r'\d+\.?\d*', gold_answer)
    answer_numbers = re.findall(r'\d+\.?\d*', answer)
    if gold_numbers and gold_numbers == answer_numbers[:len(gold_numbers)]:
        return True
    
    # 默认为不相等
    return False


def batch_qa(questions, query_chain, db, embeddings):
    """批量问答（包括检索和评估）
    
    Args:
        questions: 问题列表
        query_chain: 检索链
        db: 向量数据库
        embeddings: 嵌入模型，用于计算正确的余弦相似度
    """
    results = []
    total = len(questions)
    
    for idx, item in enumerate(questions, 1):
        question_id = item.get("id")
        question_text = item.get("question")
        gold_answer = item.get("gold_answer", "")
        
        print(f"\n{'='*80}")
        print(f"处理进度: [{idx}/{total}] - ID: {question_id}")
        print(f"问题: {question_text}")
        
        try:
            # 使用 RAG 系统生成答案
            response = query_chain.invoke({"question": question_text})
            answer = response["answer"]
            
            # 从向量数据库获取文档（Chroma 的相似度分数是欧几里得距离，需要手动计算余弦相似度）
            docs_with_scores = db.similarity_search_with_score(question_text, k=3)
            
            # 获取查询的嵌入向量
            query_embedding = np.array(embeddings.embed_query(question_text))
            
            # 提取源文档元数据和正确的余弦相似度分数
            sources = []
            for doc, _ in docs_with_scores:  # 忽略 Chroma 返回的欧几里得距离
                # 手动计算余弦相似度
                doc_embedding = np.array(embeddings.embed_query(doc.page_content))
                cosine_sim = np.dot(query_embedding, doc_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                )
                sources.append({
                    "page": doc.metadata.get("page", "N/A"),
                    "source": doc.metadata.get("source", "N/A"),
                    "similarity_score": float(cosine_sim)  # 正确的余弦相似度 (0-1)
                })
            
            # 智能判断答案是否正确
            answer_correctness = evaluate_answer_correctness(answer, gold_answer)
            
            result = {
                "id": question_id,
                "question": question_text,
                "answer": answer,
                "gold_answer": gold_answer,
                "answer_correctness": answer_correctness,  # 改进的判断
                "sources": sources
            }
            
            results.append(result)
            
            # 打印答案摘要和正确性
            answer_preview = answer[:100] + "..." if len(answer) > 100 else answer
            status = "✓" if answer_correctness else "✗"
            print(f"答案预览: {answer_preview}")
            print(f"正确性: {status}")
            print(f"相似度分数: {[round(s['similarity_score'], 3) for s in sources]}")
            
        except Exception as e:
            print(f"错误: 处理问题 ID {question_id} 时失败 - {e}")
            results.append({
                "id": question_id,
                "question": question_text,
                "answer": f"ERROR: {str(e)}",
                "gold_answer": gold_answer,
                "answer_correctness": False,
                "sources": []
            })
    
    return results


def save_results(results, output_file):
    """保存结果到 JSON 文件"""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n✅ 成功保存 {len(results)} 条结果到: {output_file}")
    except Exception as e:
        print(f"❌ 保存结果失败: {e}")


def main():
    print("="*80)
    print("批量问答脚本启动")
    print("="*80)
    
    # 1. 加载 benchmark 问题
    questions = load_benchmark_questions(BENCHMARK_FILE)
    if not questions:
        print("没有问题需要处理，程序退出。")
        return
    
    # 2. 初始化 RAG 系统
    print("\n初始化 RAG 系统...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = get_embed_db(embeddings)
    retriever = db.as_retriever()
    llm = load_llm()
    
    # 3. 创建对话链（每次批处理使用新的 memory）
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        output_key="answer",
        return_messages=True,
        window_size=MEMORY_WINDOW_SIZE,
    )
    
    query_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=memory,
        retriever=retriever,
        verbose=VERBOSE,
        return_source_documents=True,
    )
    
    # 4. 批量问答
    print("\n开始批量问答...")
    results = batch_qa(questions, query_chain, db, embeddings)
    
    # 5. 保存结果
    save_results(results, OUTPUT_FILE)
    
    print("\n" + "="*80)
    print("批量问答完成！")
    print("="*80)


if __name__ == "__main__":
    main()
