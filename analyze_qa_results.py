"""分析 QA 结果的 JSON 文件"""

import json
import sys

def analyze_qa_results(json_file):
    """分析 QA 结果"""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
    except FileNotFoundError:
        print(f"❌ 错误: 找不到文件 {json_file}")
        return
    except json.JSONDecodeError as e:
        print(f"❌ 错误: JSON 解析失败 - {e}")
        return
    
    print("="*80)
    print(f"📊 分析文件: {json_file}")
    print(f"📋 总条目数: {len(results)}")
    print("="*80)
    
    # 1. 查找没有 answer 或 answer 为错误的条目
    no_answer = []
    error_answer = []
    
    for item in results:
        answer = item.get("answer", "")
        if not answer or answer.strip() == "":
            no_answer.append(item)
        elif answer.startswith("ERROR:"):
            error_answer.append(item)
    
    # 2. 查找 answer_correctness 为空或 N/A 的条目
    missing_correctness = []
    
    for item in results:
        correctness = item.get("answer_correctness")
        if correctness is None or correctness == "" or correctness == "N/A":
            missing_correctness.append(item)
    
    # 3. 查找 similarity_score 为空或 N/A 的条目
    missing_similarity = []
    
    for item in results:
        sources = item.get("sources", [])
        if sources:
            for source in sources:
                score = source.get("similarity_score")
                if score is None or score == "" or score == "N/A":
                    missing_similarity.append(item)
                    break
        else:
            # 没有 sources 的也算作缺失
            missing_similarity.append(item)
    
    # 打印结果
    print("\n" + "="*80)
    print("🔴 1. 没有答案或答案为空的条目")
    print("="*80)
    if no_answer:
        print(f"找到 {len(no_answer)} 条:")
        for item in no_answer:
            print(f"  - ID {item['id']}: {item['question'][:60]}...")
    else:
        print("✅ 所有条目都有答案")
    
    print("\n" + "="*80)
    print("⚠️  2. 答案包含错误（ERROR）的条目（需要手工重新运行）")
    print("="*80)
    if error_answer:
        print(f"找到 {len(error_answer)} 条:")
        for item in error_answer:
            print(f"  - ID {item['id']}: {item['question'][:60]}...")
            print(f"    错误信息: {item['answer']}")
    else:
        print("✅ 没有错误答案")
    
    print("\n" + "="*80)
    print("📊 3. answer_correctness 为空或 N/A 的条目")
    print("="*80)
    if missing_correctness:
        print(f"找到 {len(missing_correctness)} 条:")
        for item in missing_correctness:
            correctness = item.get("answer_correctness")
            print(f"  - ID {item['id']}: correctness = {correctness}")
    else:
        print("✅ 所有条目都有 answer_correctness 值")
    
    print("\n" + "="*80)
    print("📈 4. similarity_score 为空或 N/A 的条目")
    print("="*80)
    if missing_similarity:
        print(f"找到 {len(missing_similarity)} 条:")
        for item in missing_similarity:
            print(f"  - ID {item['id']}: {item['question'][:60]}...")
            sources = item.get("sources", [])
            if sources:
                for i, source in enumerate(sources, 1):
                    score = source.get("similarity_score", "N/A")
                    print(f"    Source {i}: similarity_score = {score}")
            else:
                print(f"    ⚠️  没有 sources 数据")
    else:
        print("✅ 所有条目都有有效的 similarity_score")
    
    # 5. 生成需要重新运行的 ID 列表
    print("\n" + "="*80)
    print("📝 总结")
    print("="*80)
    
    need_rerun = [item['id'] for item in error_answer]
    if need_rerun:
        print(f"\n⚠️  需要手工重新运行的 ID 列表: {need_rerun}")
        print(f"   共 {len(need_rerun)} 条")
    
    all_na_similarity = len(missing_similarity)
    if all_na_similarity > 0:
        print(f"\n📊 有 {all_na_similarity} 条的 similarity_score 为 N/A")
        print("   这可能是因为检索时没有返回相似度分数")
    
    if missing_correctness:
        print(f"\n📊 有 {len(missing_correctness)} 条的 answer_correctness 为空或 N/A")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    # 默认文件名
    json_file = "qa_results_20260106_223043.json"
    
    # 如果命令行提供了参数，使用参数指定的文件
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
    
    analyze_qa_results(json_file)
