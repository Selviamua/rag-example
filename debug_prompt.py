"""Prompt 诊断脚本 - 分析生成阶段的问题
专注于 ID 5, 7：检索成功但生成失败的案例
"""

import os
import json
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferWindowMemory

COLLECTION_NAME = "doc_index"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
VERBOSE = True  # 开启详细日志


def get_embed_db(embeddings):
    """获取向量数据库"""
    chroma_persist_dir = os.getenv("CHROMA_PERSIST_DIR")
    db = Chroma(
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=chroma_persist_dir,
    )
    return db


def load_llm(temperature=0.5):
    """加载 LLM"""
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    deepseek_model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    deepseek_base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    
    return ChatOpenAI(
        model=deepseek_model,
        api_key=deepseek_api_key,
        base_url=deepseek_base_url,
        temperature=temperature,
        verbose=VERBOSE,
    )


def load_benchmark(benchmark_file):
    """加载 benchmark 数据"""
    with open(benchmark_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {item['id']: item for item in data}


def estimate_tokens(text):
    """粗略估算 token 数量（英文约为字数/4）"""
    return len(text.split()) // 4 + len(text) // 4


def analyze_prompt_generation(question_id, benchmark_data, temperature=0.5):
    """分析单个问题的 Prompt 和生成过程"""
    
    if question_id not in benchmark_data:
        print(f"❌ 问题 ID {question_id} 不存在\n")
        return
    
    item = benchmark_data[question_id]
    question = item['question']
    gold_answer = item['gold_answer']
    
    print("\n" + "="*80)
    print(f"【问题 ID: {question_id}】")
    print("="*80)
    
    # ========== 1. 基本信息 ==========
    print(f"\n【1️⃣  问题】")
    print(f"   {question}")
    
    print(f"\n【📋 金标答案】")
    print(f"   {gold_answer}")
    
    # ========== 2. 检索阶段 ==========
    print(f"\n【2️⃣  检索阶段 - Top-3 文段】")
    print("-" * 80)
    
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = get_embed_db(embeddings)
    
    docs_with_scores = db.similarity_search_with_score(question, k=3)
    
    retrieved_texts = []
    total_retrieved_length = 0
    
    for rank, (doc, _) in enumerate(docs_with_scores, 1):
        content = doc.page_content
        retrieved_texts.append(content)
        total_retrieved_length += len(content)
        
        # 计算余弦相似度
        query_emb = np.array(embeddings.embed_query(question))
        doc_emb = np.array(embeddings.embed_query(content))
        cosine_sim = np.dot(query_emb, doc_emb) / (
            np.linalg.norm(query_emb) * np.linalg.norm(doc_emb)
        )
        
        print(f"\n   【第 {rank} 名】相似度: {cosine_sim:.4f}")
        print(f"   页码: {doc.metadata.get('page', 'N/A')}")
        print(f"   文段长度: {len(content)} 字符")
        print(f"   内容预览: {content[:200]}...")
    
    print(f"\n   总检索长度: {total_retrieved_length} 字符")
    print(f"   估算 tokens: ~{estimate_tokens(''.join(retrieved_texts))}")
    
    # ========== 3. 生成阶段 ==========
    print(f"\n【3️⃣  生成阶段 - 调用 LLM】")
    print("-" * 80)
    print(f"   Temperature: {temperature}")
    
    # 创建 RAG 链
    retriever = db.as_retriever()
    llm = load_llm(temperature=temperature)
    
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        output_key="answer",
        return_messages=True,
        window_size=10,
    )
    
    query_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=memory,
        retriever=retriever,
        verbose=VERBOSE,
        return_source_documents=True,
    )
    
    print(f"\n   【调用 LLM...】")
    response = query_chain.invoke({"question": question})
    
    generated_answer = response["answer"]
    source_docs = response.get("source_documents", [])
    
    print(f"\n   【生成的答案】")
    print(f"   {generated_answer}")
    
    print(f"\n   答案长度: {len(generated_answer)} 字符")
    print(f"   估算 tokens: ~{estimate_tokens(generated_answer)}")
    
    # ========== 4. 对比分析 ==========
    print(f"\n【4️⃣  对比分析】")
    print("-" * 80)
    
    # 检查金标答案的关键信息是否在检索文段中
    gold_keywords = gold_answer.lower().split()[:10]  # 取前10个词作为关键词
    
    found_in_retrieval = []
    for kw in gold_keywords:
        for text in retrieved_texts:
            if kw in text.lower():
                found_in_retrieval.append(kw)
                break
    
    retrieval_coverage = len(found_in_retrieval) / len(gold_keywords) if gold_keywords else 0
    
    print(f"\n   【检索覆盖度】")
    print(f"   金标答案的关键词在检索文段中的覆盖: {retrieval_coverage*100:.1f}%")
    print(f"   找到的关键词: {found_in_retrieval[:5]}...")
    
    # 检查生成答案是否包含金标答案的关键信息
    answer_match = gold_answer.lower() in generated_answer.lower()
    
    print(f"\n   【生成质量】")
    if answer_match:
        print(f"   ✅ 生成答案包含金标答案")
    else:
        print(f"   ❌ 生成答案未包含金标答案")
    
    # 检查生成答案是否使用了检索内容
    uses_retrieval = False
    for text in retrieved_texts:
        # 取检索文段的特征片段
        unique_phrase = ' '.join(text.split()[:10])
        if unique_phrase.lower() in generated_answer.lower():
            uses_retrieval = True
            break
    
    if uses_retrieval:
        print(f"   ✅ 生成答案使用了检索内容")
    else:
        print(f"   ⚠️  生成答案可能未充分使用检索内容")
    
    # ========== 5. 诊断结论 ==========
    print(f"\n【🔍 诊断结论】")
    print("-" * 80)
    
    if retrieval_coverage > 0.7:
        print(f"   ✅ 检索质量优秀：文段包含 {retrieval_coverage*100:.0f}% 的答案关键词")
    elif retrieval_coverage > 0.4:
        print(f"   ⚠️  检索质量中等：文段包含 {retrieval_coverage*100:.0f}% 的答案关键词")
    else:
        print(f"   ❌ 检索质量差：文段仅包含 {retrieval_coverage*100:.0f}% 的答案关键词")
    
    if not answer_match:
        if uses_retrieval:
            print(f"\n   【诊断】：LLM 使用了检索内容，但生成了错误答案")
            print(f"   可能原因：")
            print(f"     1. 检索文段中的信息不够准确或完整")
            print(f"     2. LLM 对检索内容的理解有偏差")
            print(f"     3. Temperature={temperature} 导致生成过于创意")
            print(f"\n   【建议】：")
            print(f"     - 尝试降低 temperature (→ 0.1 或 0.0)")
            print(f"     - 优化 Prompt 模板，明确要求基于检索内容回答")
        else:
            print(f"\n   【诊断】：LLM 可能忽略了检索内容，自己编造答案")
            print(f"   可能原因：")
            print(f"     1. Prompt 模板不清晰，LLM 没理解要用检索内容")
            print(f"     2. 检索文段与问题关联性不强")
            print(f"     3. Temperature 过高导致过度创意")
            print(f"\n   【建议】：")
            print(f"     - 检查并优化 ConversationalRetrievalChain 的 system prompt")
            print(f"     - 降低 temperature")
            print(f"     - 考虑使用重排（reranker）提升检索质量")
    else:
        print(f"\n   ✅ 生成成功！答案正确。")
    
    print("\n")


def compare_temperatures(question_id, benchmark_data):
    """对比不同 temperature 的生成效果"""
    
    print("\n" + "="*80)
    print(f"【Temperature 对比实验 - ID {question_id}】")
    print("="*80)
    
    temperatures = [0.5, 0.1, 0.0]
    
    for temp in temperatures:
        print(f"\n{'─'*80}")
        print(f"【Temperature = {temp}】")
        print(f"{'─'*80}")
        
        analyze_prompt_generation(question_id, benchmark_data, temperature=temp)
        
        print("\n⏳ 等待 3 秒后继续...\n")
        import time
        time.sleep(3)


def main():
    print("\n" + "="*80)
    print("【RAG Prompt & 生成阶段诊断工具】")
    print("专注于检索成功但生成失败的案例 (ID 5, 7)")
    print("="*80)
    
    benchmark_data = load_benchmark("benchmark.json")
    
    # 重点分析的两个问题
    problem_ids = [5, 7]
    
    print(f"\n【单次诊断】")
    for qid in problem_ids:
        analyze_prompt_generation(qid, benchmark_data, temperature=0.5)
    
    # 可选：对比不同 temperature
    print(f"\n" + "="*80)
    print("是否进行 Temperature 对比实验？(需要更多时间和 API 调用)")
    print("="*80)
    # 默认不运行，如需运行可取消注释
    # compare_temperatures(5, benchmark_data)


if __name__ == "__main__":
    main()
