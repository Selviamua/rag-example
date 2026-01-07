"""诊断脚本 - 深度分析检索阶段的质量
用于排查为什么某些问题的答案错误：是检索不到相关内容？还是别的原因？
"""

import os
import json
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

COLLECTION_NAME = "doc_index"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def get_embed_db(embeddings):
    """获取向量数据库"""
    chroma_persist_dir = os.getenv("CHROMA_PERSIST_DIR")
    if not chroma_persist_dir:
        raise EnvironmentError("未找到 CHROMA_PERSIST_DIR 环境变量")
    
    db = Chroma(
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=chroma_persist_dir,
    )
    return db


def load_benchmark(benchmark_file):
    """加载 benchmark 数据"""
    with open(benchmark_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {item['id']: item for item in data}


def extract_keywords(text, num_keywords=5):
    """简单的关键词提取（不用 NLP 库，避免复杂依赖）
    取文本中较长的词作为候选关键词
    """
    # 分词
    words = text.lower().split()
    # 去掉太短的词和标点
    keywords = [w.strip('.,;:!?') for w in words if len(w.strip('.,;:!?')) > 3]
    return keywords[:num_keywords]


def keyword_in_text(keywords, text):
    """检查关键词是否在文本中出现"""
    text_lower = text.lower()
    found = []
    for kw in keywords:
        if kw.lower() in text_lower:
            found.append(kw)
    return found


def analyze_retrieval(question_id, db, benchmark_data, embeddings):
    """分析单个问题的检索效果"""
    
    if question_id not in benchmark_data:
        print(f"❌ 问题 ID {question_id} 不存在\n")
        return
    
    item = benchmark_data[question_id]
    question = item['question']
    gold_answer = item['gold_answer']
    
    print("\n" + "="*80)
    print(f"【问题 ID: {question_id}】")
    print("="*80)
    
    # ========== 1. 问题信息 ==========
    print(f"\n【1️⃣  问题文本】")
    print(f"   {question}")
    
    print(f"\n【📋 金标答案】")
    print(f"   {gold_answer}")
    
    # ========== 2. 关键词提取 ==========
    gold_keywords = extract_keywords(gold_answer, num_keywords=5)
    print(f"\n【🔑 关键词（从金标答案提取）】")
    print(f"   {gold_keywords}")
    
    # ========== 3. 向量检索 ==========
    print(f"\n【2️⃣  检索阶段 - Top-3 结果】")
    print("-" * 80)
    
    docs_with_scores = db.similarity_search_with_score(question, k=3)
    
    for rank, (doc, chroma_score) in enumerate(docs_with_scores, 1):
        # 获取文档内容和元数据
        content = doc.page_content
        metadata = doc.metadata
        page = metadata.get('page', 'N/A')
        source = metadata.get('source', 'N/A')
        
        # 手动计算余弦相似度（修正）
        query_embedding = np.array(embeddings.embed_query(question))
        doc_embedding = np.array(embeddings.embed_query(content))
        cosine_sim = np.dot(query_embedding, doc_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
        )
        
        # 检查关键词匹配
        found_keywords = keyword_in_text(gold_keywords, content)
        keyword_match_rate = len(found_keywords) / len(gold_keywords) if gold_keywords else 0
        
        print(f"\n   【第 {rank} 名】")
        print(f"   相似度分数（Cosine）: {cosine_sim:.4f}")
        print(f"   页码: {page}, 来源: {source}")
        print(f"   文档内容（前150字）:")
        print(f"   {content[:150]}...")
        print(f"   \n   关键词匹配: {found_keywords} ({keyword_match_rate*100:.0f}%)")
        
    # ========== 4. 检索质量评估 ==========
    print(f"\n【3️⃣  检索质量评估】")
    print("-" * 80)
    
    max_score = docs_with_scores[0][1] if docs_with_scores else 0
    # 修正后的最高相似度
    query_emb = np.array(embeddings.embed_query(question))
    doc_emb = np.array(embeddings.embed_query(docs_with_scores[0][0].page_content))
    max_cosine = np.dot(query_emb, doc_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(doc_emb))
    
    scores = []
    all_found_keywords = set()
    for doc, _ in docs_with_scores:
        q_emb = np.array(embeddings.embed_query(question))
        d_emb = np.array(embeddings.embed_query(doc.page_content))
        cos_sim = np.dot(q_emb, d_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(d_emb))
        scores.append(cos_sim)
        found_keywords = keyword_in_text(gold_keywords, doc.page_content)
        all_found_keywords.update(found_keywords)
    
    avg_score = np.mean(scores)
    max_score_corrected = max(scores)
    
    print(f"   最高相似度分数: {max_score_corrected:.4f}")
    print(f"   平均相似度分数: {avg_score:.4f}")
    print(f"   检索质量判定: ", end="")
    
    if max_score_corrected >= 0.55 or avg_score >= 0.52:
        print("✅ HIGH (相关内容存在)")
    else:
        print("⚠️  LOW (文档库中相关内容较少)")
    
    print(f"\n   Top-3 中找到的关键词: {list(all_found_keywords)}")
    if all_found_keywords:
        print(f"   ✓ 好消息：文档中包含答案的关键信息")
    else:
        print(f"   ✗ 坏消息：文档中可能不包含答案的关键信息")
    
    # ========== 5. 诊断结论 ==========
    print(f"\n【🔍 诊断结论】")
    print("-" * 80)
    
    if max_score_corrected < 0.55:
        print(f"   ⚠️  【检索失败】")
        print(f"      - 相似度分数较低 ({max_score_corrected:.4f} < 0.55)")
        print(f"      - 可能原因：")
        print(f"        1. 文档库中没有相关内容")
        print(f"        2. 问题表述与文档表述差异大")
        print(f"        3. 嵌入模型对该问题的理解不足")
        print(f"      - 改进建议：检查源文档是否包含答案；考虑问题改写或模型升级")
    elif all_found_keywords:
        print(f"   ✅ 【检索成功】")
        print(f"      - 相似度分数良好 ({max_score_corrected:.4f} >= 0.55)")
        print(f"      - 关键词匹配 ({len(all_found_keywords)}/{len(gold_keywords)} 个关键词)")
        print(f"      - 诊断：文档库包含正确答案")
        print(f"      - 如果最终答案还是错，可能问题在后续阶段（Prompt 或 LLM 生成）")
    else:
        print(f"   ⚠️  【部分成功】")
        print(f"      - 相似度分数中等 ({max_score_corrected:.4f})")
        print(f"      - 但关键词匹配不理想")
        print(f"      - 诊断：检索到的是相似内容，但可能不是最直接的答案")
    
    print("\n")


def main():
    print("\n" + "="*80)
    print("【RAG 检索阶段诊断工具】")
    print("用于分析检索阶段的质量，排查答案错误的原因")
    print("="*80)
    
    # 初始化
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = get_embed_db(embeddings)
    benchmark_data = load_benchmark("benchmark.json")
    
    # 需要深度分析的失败案例
    problem_ids = [4, 5, 7, 16, 18, 20, 23]
    
    print(f"\n分析对象：{len(problem_ids)} 个失败/幻觉案例")
    print(f"问题 ID: {problem_ids}")
    
    # 逐个分析
    for qid in problem_ids:
        analyze_retrieval(qid, db, benchmark_data, embeddings)
    
    # 汇总统计
    print("\n" + "="*80)
    print("【总体统计】")
    print("="*80)
    
    high_count = 0
    low_count = 0
    
    for qid in problem_ids:
        if qid not in benchmark_data:
            continue
        
        item = benchmark_data[qid]
        question = item['question']
        
        docs_with_scores = db.similarity_search_with_score(question, k=3)
        scores = []
        for doc, _ in docs_with_scores:
            q_emb = np.array(embeddings.embed_query(question))
            d_emb = np.array(embeddings.embed_query(doc.page_content))
            cos_sim = np.dot(q_emb, d_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(d_emb))
            scores.append(cos_sim)
        
        max_score = max(scores)
        avg_score = np.mean(scores)
        
        if max_score >= 0.55 or avg_score >= 0.52:
            high_count += 1
        else:
            low_count += 1
    
    print(f"\n检索质量统计 (基于 0.55/0.52 阈值):")
    print(f"  HIGH (相关内容存在): {high_count}/{len(problem_ids)}")
    print(f"  LOW  (相关内容缺少): {low_count}/{len(problem_ids)}")
    print(f"\n结论:")
    if high_count >= len(problem_ids) * 0.7:
        print(f"  ✓ 检索质量总体良好")
        print(f"  → 答案错误可能不是检索问题，需要诊断后续阶段（Prompt/LLM）")
    else:
        print(f"  ⚠️  检索质量需要改进")
        print(f"  → 建议改进：1) chunk 分割策略  2) 嵌入模型  3) 问题改写")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
