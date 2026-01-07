"""测试不同的 chunking 参数对检索质量的影响"""

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

# 测试配置
TEST_CONFIGS = [
    # 原始配置
    {"chunk_size": 512, "chunk_overlap": 0, "name": "原始 (512/0)"},
    # 小 chunk + 中等 overlap
    {"chunk_size": 256, "chunk_overlap": 50, "name": "小块 (256/50)"},
    {"chunk_size": 256, "chunk_overlap": 100, "name": "小块 (256/100)"},
    # 中等 chunk + 不同 overlap
    {"chunk_size": 512, "chunk_overlap": 50, "name": "中块 (512/50)"},
    {"chunk_size": 512, "chunk_overlap": 100, "name": "中块 (512/100)"},
    {"chunk_size": 512, "chunk_overlap": 150, "name": "中块 (512/150)"},
    # 大 chunk + 不同 overlap
    {"chunk_size": 1024, "chunk_overlap": 100, "name": "大块 (1024/100)"},
    {"chunk_size": 1024, "chunk_overlap": 200, "name": "大块 (1024/200)"},
]

# 测试问题 (ID 5 和 7)
TEST_QUESTIONS = [
    {
        "id": 5,
        "question": "What are the limitations of bioplastic based on current microorganisms?",
        "gold_answer": "Lower tolerances to harsh industrial conditions compared to their synthetic counterparts."
    },
    {
        "id": 7,
        "question": "How can extremophiles help overcome these limitations?",
        "gold_answer": "Extremophiles have adaptations that allow them to thrive in extreme conditions, which could be harnessed to produce bioplastics with improved properties."
    }
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
        separators=["\n \n", "\n\n", "\n", " ", ""],
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


def test_retrieval(db, embeddings, question, question_id):
    """测试检索质量"""
    query_embedding = embeddings.embed_query(question["question"])
    
    # 获取 Top-3 文档及其嵌入
    results = db.similarity_search_with_score(question["question"], k=3)
    
    retrieval_info = {
        "question_id": question_id,
        "top_docs": []
    }
    
    similarities = []
    for i, (doc, _) in enumerate(results):
        doc_embedding = embeddings.embed_query(doc.page_content)
        similarity = calculate_cosine_similarity(query_embedding, doc_embedding)
        similarities.append(similarity)
        
        retrieval_info["top_docs"].append({
            "rank": i + 1,
            "similarity": float(similarity),
            "page": doc.metadata.get("page", "N/A"),
            "content_preview": doc.page_content[:100] + "..."
        })
    
    retrieval_info["max_similarity"] = float(max(similarities))
    retrieval_info["avg_similarity"] = float(np.mean(similarities))
    
    return retrieval_info


def main():
    print("=" * 80)
    print("🔬 Chunking 参数优化实验")
    print("=" * 80)
    print(f"📊 测试配置数量: {len(TEST_CONFIGS)}")
    print(f"❓ 测试问题数量: {len(TEST_QUESTIONS)}")
    print()
    
    all_results = []
    
    for config_idx, config in enumerate(TEST_CONFIGS, 1):
        print(f"\n{'='*80}")
        print(f"📋 配置 {config_idx}/{len(TEST_CONFIGS)}: {config['name']}")
        print(f"   chunk_size={config['chunk_size']}, chunk_overlap={config['chunk_overlap']}")
        print(f"{'='*80}")
        
        collection_name = f"test_chunk_{config['chunk_size']}_{config['chunk_overlap']}"
        
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
            for question in TEST_QUESTIONS:
                print(f"\n  ❓ 测试问题 ID {question['id']}...")
                retrieval_info = test_retrieval(db, embeddings, question, question['id'])
                config_result["questions"].append(retrieval_info)
                
                print(f"    最高相似度: {retrieval_info['max_similarity']:.4f}")
                print(f"    平均相似度: {retrieval_info['avg_similarity']:.4f}")
                print(f"    Top-1 页码: {retrieval_info['top_docs'][0]['page']}")
            
            all_results.append(config_result)
            
        except Exception as e:
            print(f"  ❌ 配置测试失败: {e}")
            continue
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"chunking_test_results_{timestamp}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n\n{'='*80}")
    print(f"✅ 实验完成！结果已保存到: {output_file}")
    print(f"{'='*80}")
    
    # 打印对比分析
    print("\n\n📊 配置对比分析")
    print("=" * 80)
    print(f"{'配置':<20} {'文档块数':<10} {'ID 5 相似度':<15} {'ID 7 相似度':<15}")
    print("-" * 80)
    
    for result in all_results:
        config_name = result['config']['name']
        num_chunks = result['num_chunks']
        
        q5_result = next((q for q in result['questions'] if q['question_id'] == 5), None)
        q7_result = next((q for q in result['questions'] if q['question_id'] == 7), None)
        
        q5_sim = f"{q5_result['max_similarity']:.4f}" if q5_result else "N/A"
        q7_sim = f"{q7_result['max_similarity']:.4f}" if q7_result else "N/A"
        
        print(f"{config_name:<20} {num_chunks:<10} {q5_sim:<15} {q7_sim:<15}")
    
    # 找出最佳配置
    print("\n\n🏆 最佳配置推荐")
    print("=" * 80)
    
    best_for_q5 = max(all_results, 
                      key=lambda r: next((q['max_similarity'] for q in r['questions'] if q['question_id'] == 5), 0))
    best_for_q7 = max(all_results,
                      key=lambda r: next((q['max_similarity'] for q in r['questions'] if q['question_id'] == 7), 0))
    
    print(f"问题 5 最佳配置: {best_for_q5['config']['name']}")
    q5_best = next(q for q in best_for_q5['questions'] if q['question_id'] == 5)
    print(f"  相似度: {q5_best['max_similarity']:.4f}")
    print(f"  文档块数: {best_for_q5['num_chunks']}")
    
    print(f"\n问题 7 最佳配置: {best_for_q7['config']['name']}")
    q7_best = next(q for q in best_for_q7['questions'] if q['question_id'] == 7)
    print(f"  相似度: {q7_best['max_similarity']:.4f}")
    print(f"  文档块数: {best_for_q7['num_chunks']}")


if __name__ == "__main__":
    main()
