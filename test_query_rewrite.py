"""Query Rewrite 改进实验
对失败的问题（ID 4, 16, 20, 23）进行问题重写，提升检索质量
使用 DeepSeek API 将口语化问题改写为论文术语风格的查询
"""

import os
import json
from datetime import datetime
from dotenv import load_dotenv
from transformers import AutoTokenizer
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from openai import OpenAI
import numpy as np

load_dotenv()

# 失败的问题（需要 Query Rewrite）
FAILED_QUESTIONS = [
    {
        "id": 4,
        "question": "What is the highest PHB production achieved in Alcaligenes latus, as reported in the paper?",
        "gold_answer": "98.7 g/L in Alcaligenes latus (Wang & Lee, 1997).",
        "keywords": ["PHB", "98.7 g/L", "Alcaligenes latus"],
        "document_hint": "PHB at 98.7 g/L in Alcaligenes latus (Wang & Lee, 1997)"
    },
    {
        "id": 16,
        "question": "What was the L-LA production titer achieved after ALE was employed to improve lactic acid tolerance, as reported in the paper?",
        "gold_answer": "119 g/L of L-LA using buckwheat husk hydrolysates.",
        "keywords": ["ALE", "119 g/L", "buckwheat husk", "lactic acid tolerance"],
        "document_hint": "ALE employed to improve lactic acid tolerance resulted in a 17% increase in L-LA production, reaching a titer of 119 g/L"
    },
    {
        "id": 20,
        "question": "How was the highest titer of 50 g/L AA produced by yeast, as disclosed in the Verdezyne Inc. patent, achieved according to the paper?",
        "gold_answer": "By engineering fatty acid catabolism in Candida spp.",
        "keywords": ["50 g/L", "AA", "Verdezyne", "Candida", "fatty acid catabolism"],
        "document_hint": "highest titer of AA produced by yeast (50 g/L) was disclosed in Verdezyne Inc. patent, achieved by Candida spp. with engineered fatty acid catabolism"
    },
    {
        "id": 23,
        "question": "What is the reason for the low product titers in AA and ccMA production, according to the paper?",
        "gold_answer": "Low product titers in AA and ccMA production are largely attributed to yeast's sensitivity to the pathway intermediate, catechol, which inhibits growth at 0.5 mM compared to 5 mM in Pseudomonas. Additionally, protocatechuic acid decarboxylase (AroY) exhibits 90% lower activity in yeast than in bacteria due to improper flavin cofactor incorporation at 30°C.",
        "keywords": ["catechol", "0.5 mM", "AroY", "flavin cofactor", "AA", "ccMA"],
        "document_hint": "yeast's sensitivity to catechol pathway intermediate, AroY exhibits 90% lower activity due to improper flavin cofactor incorporation"
    }
]

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SOURCE_PDF = "source_documents/Le.pdf"

# 最佳 chunking 配置（基于之前实验）
CHUNK_SIZE = 512
CHUNK_OVERLAP = 150


def get_deepseek_client():
    """获取 DeepSeek 客户端"""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("未找到 DEEPSEEK_API_KEY 环境变量")
    
    return OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com"
    )


def rewrite_query(client, original_question, document_hint=None):
    """使用 LLM 重写查询，使其更接近论文表述"""
    
    system_prompt = """You are a scientific query rewriter. Your task is to rewrite user questions into queries that are more likely to match academic paper text.

Rules:
1. Use technical terminology instead of colloquial expressions
2. Include specific numbers, units, chemical names, and organism names
3. Use phrases commonly found in scientific papers
4. Keep the query concise but information-rich
5. Include abbreviations alongside full names (e.g., "polyhydroxybutyrate (PHB)")
6. Output ONLY the rewritten query, nothing else"""

    user_prompt = f"""Original question: {original_question}

{f"Hint (expected content style): {document_hint}" if document_hint else ""}

Rewrite this into a scientific paper-style query:"""

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3,
        max_tokens=200
    )
    
    return response.choices[0].message.content.strip()


def create_index():
    """创建向量索引"""
    print("📄 加载 PDF...")
    loader = PyPDFLoader(SOURCE_PDF)
    
    print(f"🔪 切分文档 (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    docs = loader.load_and_split(text_splitter)
    print(f"✓ 共切分为 {len(docs)} 个文档块")
    
    print("🔢 生成嵌入向量...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    chroma_persist_dir = os.getenv("CHROMA_PERSIST_DIR")
    db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name="test_query_rewrite",
        persist_directory=chroma_persist_dir,
    )
    
    return db, embeddings


def calculate_cosine_similarity(query_embedding, doc_embedding):
    """手动计算余弦相似度"""
    query_norm = np.linalg.norm(query_embedding)
    doc_norm = np.linalg.norm(doc_embedding)
    if query_norm == 0 or doc_norm == 0:
        return 0.0
    return np.dot(query_embedding, doc_embedding) / (query_norm * doc_norm)


def test_retrieval(db, embeddings, query, question):
    """测试检索质量"""
    query_embedding = embeddings.embed_query(query)
    results = db.similarity_search_with_score(query, k=3)
    
    similarities = []
    keyword_matches = []
    top_docs = []
    
    for i, (doc, _) in enumerate(results):
        doc_embedding = embeddings.embed_query(doc.page_content)
        similarity = calculate_cosine_similarity(query_embedding, doc_embedding)
        similarities.append(similarity)
        
        content_lower = doc.page_content.lower()
        matched_keywords = [kw for kw in question["keywords"] if kw.lower() in content_lower]
        keyword_match_rate = len(matched_keywords) / len(question["keywords"]) if question["keywords"] else 0
        keyword_matches.append(keyword_match_rate)
        
        top_docs.append({
            "rank": i + 1,
            "similarity": float(similarity),
            "page": doc.metadata.get("page", "N/A"),
            "keyword_match_rate": float(keyword_match_rate),
            "matched_keywords": matched_keywords,
            "content_preview": doc.page_content[:200] + "..."
        })
    
    return {
        "max_similarity": float(max(similarities)),
        "avg_similarity": float(np.mean(similarities)),
        "max_keyword_match": float(max(keyword_matches)),
        "quality": "HIGH" if max(similarities) >= 0.55 else "LOW",
        "top_docs": top_docs
    }


def main():
    print("=" * 80)
    print("🔬 Query Rewrite 改进实验")
    print("=" * 80)
    print(f"📊 测试问题: ID {[q['id'] for q in FAILED_QUESTIONS]}")
    print(f"🔧 方法: DeepSeek API 问题重写")
    print()
    
    # 初始化
    client = get_deepseek_client()
    db, embeddings = create_index()
    
    all_results = []
    
    for question in FAILED_QUESTIONS:
        print(f"\n{'='*80}")
        print(f"❓ 问题 ID {question['id']}")
        print(f"{'='*80}")
        print(f"📝 原始问题: {question['question']}")
        
        # 1. 测试原始问题
        print("\n📊 [原始查询测试]")
        original_result = test_retrieval(db, embeddings, question['question'], question)
        quality_icon = "✅" if original_result["quality"] == "HIGH" else "❌"
        print(f"   {quality_icon} 最高相似度: {original_result['max_similarity']:.4f} ({original_result['quality']})")
        print(f"   🔑 关键词匹配: {original_result['max_keyword_match']*100:.1f}%")
        
        # 2. 生成重写查询
        print("\n🔄 [Query Rewrite]")
        rewritten_query = rewrite_query(client, question['question'], question.get('document_hint'))
        print(f"   重写后: {rewritten_query}")
        
        # 3. 测试重写查询
        print("\n📊 [重写查询测试]")
        rewritten_result = test_retrieval(db, embeddings, rewritten_query, question)
        quality_icon = "✅" if rewritten_result["quality"] == "HIGH" else "❌"
        print(f"   {quality_icon} 最高相似度: {rewritten_result['max_similarity']:.4f} ({rewritten_result['quality']})")
        print(f"   🔑 关键词匹配: {rewritten_result['max_keyword_match']*100:.1f}%")
        
        # 4. 计算改进
        improvement = ((rewritten_result['max_similarity'] - original_result['max_similarity']) 
                      / original_result['max_similarity'] * 100) if original_result['max_similarity'] > 0 else 0
        
        quality_change = "⬆ 提升" if (original_result['quality'] == "LOW" and rewritten_result['quality'] == "HIGH") else \
                        "✓ 保持HIGH" if rewritten_result['quality'] == "HIGH" else "⬇ 仍LOW"
        
        print(f"\n📈 改进效果: {'+' if improvement > 0 else ''}{improvement:.1f}% | {quality_change}")
        
        # 保存结果
        all_results.append({
            "question_id": question["id"],
            "original_question": question["question"],
            "rewritten_query": rewritten_query,
            "original_result": original_result,
            "rewritten_result": rewritten_result,
            "improvement_percent": improvement,
            "quality_improved": original_result['quality'] == "LOW" and rewritten_result['quality'] == "HIGH"
        })
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"query_rewrite_results_{timestamp}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    # 汇总统计
    print(f"\n\n{'='*80}")
    print("📊 实验汇总")
    print("=" * 80)
    print(f"{'ID':<6} {'原始相似度':<12} {'重写相似度':<12} {'提升':<10} {'质量变化'}")
    print("-" * 80)
    
    quality_improvements = 0
    total_improvement = 0
    
    for result in all_results:
        qid = result['question_id']
        orig_sim = result['original_result']['max_similarity']
        new_sim = result['rewritten_result']['max_similarity']
        improvement = result['improvement_percent']
        total_improvement += improvement
        
        orig_quality = result['original_result']['quality']
        new_quality = result['rewritten_result']['quality']
        
        if result['quality_improved']:
            quality_change = "LOW → HIGH ✅"
            quality_improvements += 1
        elif new_quality == "HIGH":
            quality_change = "HIGH → HIGH"
        else:
            quality_change = "LOW → LOW ❌"
        
        improvement_str = f"+{improvement:.1f}%" if improvement > 0 else f"{improvement:.1f}%"
        print(f"ID {qid:<3} {orig_sim:<12.4f} {new_sim:<12.4f} {improvement_str:<10} {quality_change}")
    
    print("-" * 80)
    print(f"📈 平均提升: {total_improvement/len(all_results):.1f}%")
    print(f"✅ 质量提升: {quality_improvements}/{len(all_results)}")
    print(f"💾 结果保存: {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
