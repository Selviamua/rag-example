"""测试改进效果 - 对比原始版本和改进版本
专注于 ID 5, 7 两个问题
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

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


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
        verbose=False,
    )


def get_db(collection_name):
    """获取指定的向量数据库"""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    chroma_persist_dir = os.getenv("CHROMA_PERSIST_DIR")
    
    db = Chroma(
        embedding_function=embeddings,
        collection_name=collection_name,
        persist_directory=chroma_persist_dir,
    )
    return db, embeddings


def load_benchmark(benchmark_file):
    """加载 benchmark 数据"""
    with open(benchmark_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {item['id']: item for item in data}


def test_single_question(question_id, collection_name, temperature, benchmark_data):
    """测试单个问题"""
    
    item = benchmark_data[question_id]
    question = item['question']
    gold_answer = item['gold_answer']
    
    print(f"\n{'─'*80}")
    print(f"【配置】Collection: {collection_name}, Temperature: {temperature}")
    print(f"{'─'*80}")
    
    # 获取数据库和嵌入模型
    db, embeddings = get_db(collection_name)
    
    # 检索阶段
    print(f"\n【检索阶段】")
    docs_with_scores = db.similarity_search_with_score(question, k=3)
    
    # 计算余弦相似度
    query_emb = np.array(embeddings.embed_query(question))
    scores = []
    
    for rank, (doc, _) in enumerate(docs_with_scores, 1):
        doc_emb = np.array(embeddings.embed_query(doc.page_content))
        cosine_sim = np.dot(query_emb, doc_emb) / (
            np.linalg.norm(query_emb) * np.linalg.norm(doc_emb)
        )
        scores.append(cosine_sim)
        
        print(f"  Top-{rank}: 相似度 {cosine_sim:.4f}, 页码 {doc.metadata.get('page', 'N/A')}")
        print(f"         内容: {doc.page_content[:80]}...")
    
    max_score = max(scores)
    avg_score = np.mean(scores)
    
    print(f"\n  最高分: {max_score:.4f}")
    print(f"  平均分: {avg_score:.4f}")
    print(f"  质量: {'✅ HIGH' if max_score >= 0.55 else '⚠️ LOW'}")
    
    # 生成阶段
    print(f"\n【生成阶段】")
    
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
        verbose=False,
        return_source_documents=True,
    )
    
    response = query_chain.invoke({"question": question})
    generated_answer = response["answer"]
    
    print(f"  答案长度: {len(generated_answer)} 字符")
    print(f"  答案预览: {generated_answer[:200]}...")
    
    # 评估
    print(f"\n【评估】")
    
    # 简单的包含判断
    answer_correct = gold_answer.lower() in generated_answer.lower()
    
    print(f"  金标答案: {gold_answer[:100]}...")
    print(f"  包含金标: {'✅ Yes' if answer_correct else '❌ No'}")
    
    return {
        "question_id": question_id,
        "collection": collection_name,
        "temperature": temperature,
        "max_similarity": max_score,
        "avg_similarity": avg_score,
        "answer_correct": answer_correct,
        "answer_preview": generated_answer[:200]
    }


def compare_configurations(question_id, benchmark_data):
    """对比不同配置"""
    
    print("\n" + "="*80)
    print(f"【对比实验 - 问题 ID {question_id}】")
    print("="*80)
    
    item = benchmark_data[question_id]
    print(f"\n问题: {item['question']}")
    print(f"金标: {item['gold_answer'][:100]}...")
    
    # 配置组合
    configs = [
        ("doc_index", 0.5, "原始版本 (overlap=0, temp=0.5)"),
        ("doc_index_v2", 0.5, "改进索引 (overlap=50, temp=0.5)"),
        ("doc_index", 0.1, "原始索引 + 低温度 (overlap=0, temp=0.1)"),
        ("doc_index_v2", 0.1, "改进索引 + 低温度 (overlap=50, temp=0.1)"),
    ]
    
    results = []
    
    for collection, temp, description in configs:
        print(f"\n{'='*80}")
        print(f"【实验】{description}")
        print(f"{'='*80}")
        
        try:
            result = test_single_question(question_id, collection, temp, benchmark_data)
            result['description'] = description
            results.append(result)
        except Exception as e:
            print(f"❌ 错误: {e}")
            results.append({
                "description": description,
                "error": str(e)
            })
        
        # 间隔，避免 API 限流
        import time
        time.sleep(2)
    
    return results


def summarize_results(all_results):
    """汇总结果"""
    
    print("\n" + "="*80)
    print("【实验结果汇总】")
    print("="*80)
    
    for qid, results in all_results.items():
        print(f"\n【问题 ID {qid}】")
        print("-" * 80)
        
        for r in results:
            if 'error' in r:
                print(f"  {r['description']}: ❌ {r['error']}")
            else:
                print(f"  {r['description']}:")
                print(f"    相似度: {r['max_similarity']:.4f} (avg: {r['avg_similarity']:.4f})")
                print(f"    答案正确: {'✅' if r['answer_correct'] else '❌'}")
    
    # 找出最佳配置
    print("\n" + "="*80)
    print("【最佳配置】")
    print("="*80)
    
    for qid, results in all_results.items():
        print(f"\n问题 ID {qid}:")
        
        valid_results = [r for r in results if 'error' not in r]
        if valid_results:
            # 按答案正确性和相似度排序
            best = max(valid_results, key=lambda x: (x['answer_correct'], x['max_similarity']))
            print(f"  ✓ {best['description']}")
            print(f"    相似度: {best['max_similarity']:.4f}")
            print(f"    答案正确: {'✅' if best['answer_correct'] else '❌'}")


def main():
    print("\n" + "="*80)
    print("【RAG 改进效果测试】")
    print("测试两个问题 (ID 5, 7) 在不同配置下的表现")
    print("="*80)
    
    benchmark_data = load_benchmark("benchmark.json")
    
    # 测试问题
    problem_ids = [5, 7]
    
    all_results = {}
    
    for qid in problem_ids:
        results = compare_configurations(qid, benchmark_data)
        all_results[qid] = results
    
    # 汇总结果
    summarize_results(all_results)
    
    # 保存结果
    output_file = f"improvement_test_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ 结果已保存到: {output_file}")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
