"""重试脚本 - 处理失败的 5 条问题 (ID: 7, 8, 10, 11, 22)"""

import os
import json
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
MEMORY_WINDOW_SIZE = 10

# 需要重试的 ID
RETRY_IDS = [7, 8, 10, 11, 22]

# 原始结果文件和输出文件
ORIGINAL_RESULTS_FILE = "qa_results_20260106_223043.json"
OUTPUT_FILE = f"qa_results_complete_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"


def evaluate_answer_correctness(answer, gold_answer):
    """评估答案是否正确"""
    answer = answer.strip()
    gold_answer = gold_answer.strip()
    
    if answer == gold_answer:
        return True
    
    if gold_answer in answer:
        return True
    
    import re
    gold_numbers = re.findall(r'\d+\.?\d*', gold_answer)
    answer_numbers = re.findall(r'\d+\.?\d*', answer)
    if gold_numbers and gold_numbers == answer_numbers[:len(gold_numbers)]:
        return True
    
    return False


def load_llm():
    """加载语言模型"""
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
        db = Chroma(
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=chroma_persist_dir,
        )
    elif opensearch_url:
        username = os.getenv("OPENSEARCH_USERNAME")
        password = os.getenv("OPENSEARCH_PASSWORD")
        db = OpenSearchVectorSearch(
            embedding_function=embeddings,
            index_name=COLLECTION_NAME,
            opensearch_url=opensearch_url,
            http_auth=(username, password),
            use_ssl=False,
            verify_certs=False,
            ssl_assert_hostname=False,
            ssl_show_warn=False,
        )
    elif postgres_conn:
        db = PGVector(
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME,
            connection_string=postgres_conn,
        )
    else:
        raise EnvironmentError("No vector store environment variables found.")
    return db


def load_original_results(filename):
    """加载原始结果"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ 错误: 找不到文件 {filename}")
        return []
    except json.JSONDecodeError as e:
        print(f"❌ 错误: JSON 解析失败 - {e}")
        return []


def retry_failed_questions(benchmark_file, query_chain, db, original_results):
    """重新运行失败的问题"""
    # 从原始结果中提取失败的问题信息
    benchmark_data = load_original_results(benchmark_file)
    
    retry_results = {}  # {id: result_dict}
    
    for item in benchmark_data:
        item_id = item.get("id")
        if item_id not in RETRY_IDS:
            continue
        
        question_text = item.get("question")
        gold_answer = item.get("gold_answer", "")
        
        print(f"\n{'='*80}")
        print(f"🔄 重试 ID: {item_id}")
        print(f"问题: {question_text}")
        
        try:
            # 使用 RAG 系统生成答案
            response = query_chain.invoke({"question": question_text})
            answer = response["answer"]
            
            # 获取相似度分数
            docs_with_scores = db.similarity_search_with_score(question_text, k=3)
            
            sources = []
            for doc, score in docs_with_scores:
                sources.append({
                    "page": doc.metadata.get("page", "N/A"),
                    "source": doc.metadata.get("source", "N/A"),
                    "similarity_score": float(score)
                })
            
            # 评估答案
            answer_correctness = evaluate_answer_correctness(answer, gold_answer)
            
            result = {
                "id": item_id,
                "question": question_text,
                "answer": answer,
                "gold_answer": gold_answer,
                "answer_correctness": answer_correctness,
                "sources": sources
            }
            
            retry_results[item_id] = result
            
            # 打印结果
            answer_preview = answer[:100] + "..." if len(answer) > 100 else answer
            status = "✓" if answer_correctness else "✗"
            print(f"✅ 成功重试")
            print(f"答案预览: {answer_preview}")
            print(f"正确性: {status}")
            print(f"相似度分数: {[round(s['similarity_score'], 3) for s in sources]}")
            
        except Exception as e:
            print(f"❌ 错误: 重试失败 - {e}")
            retry_results[item_id] = {
                "id": item_id,
                "question": question_text,
                "answer": f"ERROR: {str(e)}",
                "gold_answer": gold_answer,
                "answer_correctness": False,
                "sources": []
            }
    
    return retry_results


def merge_results(original_results, retry_results):
    """合并原始结果和重试结果"""
    merged = []
    
    for item in original_results:
        item_id = item.get("id")
        if item_id in retry_results:
            # 用重试结果替换
            merged.append(retry_results[item_id])
        else:
            # 保持原始结果
            merged.append(item)
    
    return merged


def main():
    print("="*80)
    print("🔄 重试失败问题脚本启动")
    print("="*80)
    
    # 1. 加载原始结果
    print(f"\n1️⃣  加载原始结果: {ORIGINAL_RESULTS_FILE}")
    original_results = load_original_results(ORIGINAL_RESULTS_FILE)
    if not original_results:
        print("❌ 无法加载原始结果，程序退出。")
        return
    print(f"✅ 成功加载 {len(original_results)} 条结果")
    
    # 2. 初始化 RAG 系统
    print("\n2️⃣  初始化 RAG 系统...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = get_embed_db(embeddings)
    retriever = db.as_retriever()
    llm = load_llm()
    
    # 3. 创建对话链
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
    
    # 4. 重试失败的问题
    print(f"\n3️⃣  重试失败的问题 (ID: {RETRY_IDS})...")
    benchmark_file = "benchmark.json"
    retry_results = retry_failed_questions(benchmark_file, query_chain, db, original_results)
    
    # 5. 合并结果
    print("\n4️⃣  合并原始结果和重试结果...")
    merged_results = merge_results(original_results, retry_results)
    
    # 6. 保存最终结果
    print(f"\n5️⃣  保存最终结果到: {OUTPUT_FILE}")
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(merged_results, f, ensure_ascii=False, indent=2)
        print(f"✅ 成功保存完整结果!")
        print(f"   总条数: {len(merged_results)}")
        print(f"   成功重试: {len(retry_results)} 条")
        print(f"   保存位置: {OUTPUT_FILE}")
    except Exception as e:
        print(f"❌ 保存失败: {e}")
    
    print("\n" + "="*80)
    print("✅ 重试脚本完成！")
    print("="*80)


if __name__ == "__main__":
    main()
