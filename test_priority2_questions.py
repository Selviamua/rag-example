"""测试优先级2问题（检索质量差：ID 4, 16, 18, 20, 23）
目标：评估最佳 chunking 配置对低相似度问题的改善效果
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
import numpy as np

load_dotenv()

# 优先级2问题（检索质量差）
PRIORITY2_QUESTIONS = [
    {
        "id": 4,
        "question": "What is the highest PHB production achieved in Alcaligenes latus, as reported in the paper?",
        "gold_answer": "98.7 g/L in Alcaligenes latus (Wang & Lee, 1997).",
        "keywords": ["PHB", "98.7 g/L", "Alcaligenes latus"]
    },
    {
        "id": 16,
        "question": "What was the L-LA production titer achieved after ALE was employed to improve lactic acid tolerance, as reported in the paper?",
        "gold_answer": "119 g/L of L-LA using buckwheat husk hydrolysates.",
        "keywords": ["ALE", "119 g/L", "buckwheat husk"]
    },
    {
        "id": 18,
        "question": "What D-LA titer was achieved by using the GMES strategy with random integration of 13 genes from Leuconostoc mesenteroides into S. cerevisiae?",
        "gold_answer": "33.9 g/L of D-LA.",
        "keywords": ["GMES", "33.9 g/L", "Leuconostoc mesenteroides"]
    },
    {
        "id": 20,
        "question": "How was the highest titer of 50 g/L AA produced by yeast, as disclosed in the Verdezyne Inc. patent, achieved according to the paper?",
        "gold_answer": "By engineering fatty acid catabolism in Candida spp.",
        "keywords": ["50 g/L", "AA", "Verdezyne", "Candida"]
    },
    {
        "id": 23,
        "question": "What is the reason for the low product titers in AA and ccMA production, according to the paper?",
        "gold_answer": "Low product titers in AA and ccMA production are largely attributed to yeast's sensitivity to the pathway intermediate, catechol, which inhibits growth at 0.5 mM compared to 5 mM in Pseudomonas. Additionally, protocatechuic acid decarboxylase (AroY) exhibits 90% lower activity in yeast than in bacteria due to improper flavin cofactor incorporation at 30°C.",
        "keywords": ["catechol", "0.5 mM", "AroY", "flavin cofactor"]
    }
]

# 测试配置
TEST_CONFIGS = [
    {"chunk_size": 512, "chunk_overlap": 0, "name": "原始 (512/0)"},
    {"chunk_size": 512, "chunk_overlap": 150, "name": "最佳 (512/150)"},  # 基于之前实验的最佳配置
]

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SOURCE_PDF = "source_documents/Le.pdf"


def create_index_with_params(chunk_size, chunk_overlap, collection_name):
    """使用指定参数创建向量索引"""
    print(f"  📄 加载 PDF...")
    loader = PyPDFLoader(SOURCE_PDF)
    
    print(f"  🔪 切分文档 (size={chunk_size}, overlap={chunk_overlap})...")
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    docs = loader.load_and_split(text_splitter)
    print(f"  ✓ 共切分为 {len(docs)} 个文档块")
    
    print(f"  🔢 生成嵌入向量...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    chroma_persist_dir = os.getenv("CHROMA_PERSIST_DIR")
    db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=chroma_persist_dir,
    )
    db.persist()
    
    return db, embeddings, len(docs)


def calculate_cosine_similarity(query_embedding, doc_embedding):
    """手动计算余弦相似度"""
    query_norm = np.linalg.norm(query_embedding)
    doc_norm = np.linalg.norm(doc_embedding)
    if query_norm == 0 or doc_norm == 0:
        return 0.0
    return np.dot(query_embedding, doc_embedding) / (query_norm * doc_norm)


def test_retrieval(db, embeddings, question):
    """测试检索质量"""
    query_embedding = embeddings.embed_query(question["question"])
    
    # 获取 Top-3 文档及其嵌入
    results = db.similarity_search_with_score(question["question"], k=3)
    
    retrieval_info = {
        "question_id": question["id"],
        "question": question["question"],
        "gold_answer": question["gold_answer"],
        "keywords": question["keywords"],
        "top_docs": []
    }
    
    similarities = []
    keyword_matches = []
    
    for i, (doc, _) in enumerate(results):
        doc_embedding = embeddings.embed_query(doc.page_content)
        similarity = calculate_cosine_similarity(query_embedding, doc_embedding)
        similarities.append(similarity)
        
        # 检查关键词匹配
        content_lower = doc.page_content.lower()
        matched_keywords = [kw for kw in question["keywords"] if kw.lower() in content_lower]
        keyword_match_rate = len(matched_keywords) / len(question["keywords"]) if question["keywords"] else 0
        keyword_matches.append(keyword_match_rate)
        
        retrieval_info["top_docs"].append({
            "rank": i + 1,
            "similarity": float(similarity),
            "page": doc.metadata.get("page", "N/A"),
            "keyword_match_rate": float(keyword_match_rate),
            "matched_keywords": matched_keywords,
            "content_preview": doc.page_content[:150] + "..."
        })
    
    retrieval_info["max_similarity"] = float(max(similarities))
    retrieval_info["avg_similarity"] = float(np.mean(similarities))
    retrieval_info["max_keyword_match"] = float(max(keyword_matches))
    
    # 判断检索质量
    retrieval_info["quality"] = "HIGH" if retrieval_info["max_similarity"] >= 0.55 else "LOW"
    
    return retrieval_info


def main():
    print("=" * 80)
    print("🔬 优先级2问题检索质量测试")
    print("=" * 80)
    print(f"📊 测试问题数量: {len(PRIORITY2_QUESTIONS)}")
    print(f"🎯 问题 ID: {[q['id'] for q in PRIORITY2_QUESTIONS]}")
    print(f"📋 测试配置数量: {len(TEST_CONFIGS)}")
    print()
    
    all_results = []
    
    for config_idx, config in enumerate(TEST_CONFIGS, 1):
        print(f"\n{'='*80}")
        print(f"📋 配置 {config_idx}/{len(TEST_CONFIGS)}: {config['name']}")
        print(f"   chunk_size={config['chunk_size']}, chunk_overlap={config['chunk_overlap']}")
        print(f"{'='*80}")
        
        collection_name = f"test_priority2_{config['chunk_size']}_{config['chunk_overlap']}"
        
        try:
            # 创建索引
            db, embeddings, num_chunks = create_index_with_params(
                config['chunk_size'],
                config['chunk_overlap'],
                collection_name
            )
            
            config_result = {
                "config": config,
                "num_chunks": num_chunks,
                "questions": []
            }
            
            # 测试每个问题
            for question in PRIORITY2_QUESTIONS:
                print(f"\n  ❓ 问题 ID {question['id']}: {question['question'][:60]}...")
                retrieval_info = test_retrieval(db, embeddings, question)
                config_result["questions"].append(retrieval_info)
                
                quality_icon = "✅" if retrieval_info["quality"] == "HIGH" else "❌"
                print(f"    {quality_icon} 最高相似度: {retrieval_info['max_similarity']:.4f} ({retrieval_info['quality']})")
                print(f"    📊 平均相似度: {retrieval_info['avg_similarity']:.4f}")
                print(f"    🔑 关键词匹配: {retrieval_info['max_keyword_match']*100:.1f}%")
                print(f"    📄 Top-1 页码: {retrieval_info['top_docs'][0]['page']}")
            
            all_results.append(config_result)
            
        except Exception as e:
            print(f"  ❌ 配置测试失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"priority2_test_results_{timestamp}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n\n{'='*80}")
    print(f"✅ 实验完成！结果已保存到: {output_file}")
    print(f"{'='*80}")
    
    # 打印对比分析
    print("\n\n📊 配置对比分析")
    print("=" * 80)
    print(f"{'配置':<20} {'文档块数':<10} {'ID 4':<10} {'ID 16':<10} {'ID 18':<10} {'ID 20':<10} {'ID 23':<10}")
    print("-" * 80)
    
    for result in all_results:
        config_name = result['config']['name']
        num_chunks = result['num_chunks']
        
        row = f"{config_name:<20} {num_chunks:<10}"
        for qid in [4, 16, 18, 20, 23]:
            q_result = next((q for q in result['questions'] if q['question_id'] == qid), None)
            if q_result:
                sim = q_result['max_similarity']
                quality = "✅" if q_result['quality'] == "HIGH" else "❌"
                row += f" {sim:.4f}{quality:<4}"
            else:
                row += f" {'N/A':<10}"
        print(row)
    
    # 改进效果统计
    print("\n\n📈 改进效果统计")
    print("=" * 80)
    
    if len(all_results) >= 2:
        baseline = all_results[0]
        improved = all_results[1]
        
        print(f"{'问题 ID':<10} {'原始相似度':<15} {'改进相似度':<15} {'提升':<12} {'质量变化'}")
        print("-" * 80)
        
        improvements = []
        quality_changes = []
        
        for qid in [4, 16, 18, 20, 23]:
            baseline_q = next((q for q in baseline['questions'] if q['question_id'] == qid), None)
            improved_q = next((q for q in improved['questions'] if q['question_id'] == qid), None)
            
            if baseline_q and improved_q:
                baseline_sim = baseline_q['max_similarity']
                improved_sim = improved_q['max_similarity']
                improvement = ((improved_sim - baseline_sim) / baseline_sim * 100) if baseline_sim > 0 else 0
                improvements.append(improvement)
                
                baseline_quality = baseline_q['quality']
                improved_quality = improved_q['quality']
                
                if baseline_quality == "LOW" and improved_quality == "HIGH":
                    quality_change = "LOW → HIGH ⬆"
                    quality_changes.append(1)
                elif baseline_quality == improved_quality:
                    quality_change = f"{baseline_quality} (不变)"
                    quality_changes.append(0)
                else:
                    quality_change = f"{baseline_quality} → {improved_quality}"
                    quality_changes.append(-1)
                
                improvement_str = f"+{improvement:.1f}%" if improvement > 0 else f"{improvement:.1f}%"
                print(f"ID {qid:<7} {baseline_sim:<15.4f} {improved_sim:<15.4f} {improvement_str:<12} {quality_change}")
        
        print("\n" + "=" * 80)
        print(f"📊 平均改进: {np.mean(improvements):.2f}%")
        print(f"✅ 质量提升数量: {sum(1 for x in quality_changes if x > 0)}/{len(quality_changes)}")
        print(f"❌ 仍然 LOW 质量: {sum(1 for q in improved['questions'] if q['quality'] == 'LOW')}/{len(improved['questions'])}")


if __name__ == "__main__":
    main()
