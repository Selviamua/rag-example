# RAG 系统评测与优化实验

基于检索增强生成（RAG）的学术论文问答系统，包含完整的诊断工具链和优化实验。

## 📋 项目概述

本项目以生物塑料领域的学术论文为数据源，构建 RAG 问答系统，并通过系统性实验优化检索质量。核心贡献包括：

- **诊断工具链**：分离检索/生成阶段的失败诊断
- **Chunking 优化**：8种配置的系统对比实验
- **Query Rewrite**：使用 LLM 重写查询，成功率 100%，平均提升 45.4%

## 🏗️ 系统架构

```
PDF → 文档分割 → 向量化 → Chroma 存储 → 检索 → LLM 生成
         ↑                              ↑
    (chunk_size,              (Query Rewrite)
     chunk_overlap)
```

**技术栈**：
- **嵌入模型**：all-MiniLM-L6-v2 (384维)
- **向量数据库**：Chroma
- **LLM**：DeepSeek API
- **框架**：LangChain

## � 系统界面

<img src="images/ui_picture.png" alt="系统交互界面 - Streamlit UI" width="700"/>

## �📊 实验结果

### 问题诊断

对 23 个基准问题进行评测，发现 7 个失败案例，分为两类：

| 问题类型 | 问题 ID | 原因 |
|---------|--------|------|
| 检索质量差 | 4, 16, 18, 20, 23 | 相似度 < 0.55 |
| 检索到参考文献 | 5, 7 | 检索到 References 而非正文 |

### 优化效果

#### Chunking 参数优化

测试 8 种配置（chunk_size: 256/512/1024, overlap: 0~200）：

| 配置 | ID 5 相似度 | 提升 |
|------|------------|------|
| 原始 (512/0) | 0.6763 | - |
| **最佳 (512/150)** | **0.7365** | **+8.9%** |

#### Query Rewrite（核心改进）

使用 DeepSeek API 将用户问题重写为论文术语风格：

| ID | 原始相似度 | 重写后 | 提升 |
|----|-----------|--------|------|
| 4 | 0.4224 | **0.5863** | +38.8% |
| 16 | 0.4780 | **0.6051** | +26.6% |
| 20 | 0.4671 | **0.6886** | +47.4% |
| 23 | 0.3948 | **0.6668** | +68.9% |

**成功率：4/4 (100%)，平均提升 +45.4%**

### 重写示例

```
原始: "What is the highest PHB production in Alcaligenes latus?"
重写: "Maximum polyhydroxybutyrate (PHB) concentration Alcaligenes latus fermentation"

原始: "reason for low titers in AA and ccMA production"
重写: "AA ccMA low titer catechol AroY flavin cofactor"
```

**重写原则**：保留术语（全称+缩写）、数值、关键实体；删除冗余连接词。

## 📁 项目结构

```
├── 核心代码
│   ├── index_documents.py          # 文档向量化
│   ├── index_documents_improved.py # 改进版（overlap=150）
│   ├── batch_qa.py                 # 批量问答
│   ├── document_chatbot.py         # 交互式问答
│   └── document_chatbot_ui.py      # Streamlit UI
│
├── 诊断工具
│   ├── debug_retrieval.py          # 检索阶段诊断
│   ├── debug_prompt.py             # 生成阶段诊断
│   └── similarity_explanation.py   # 相似度计算说明
│
├── 实验脚本
│   ├── test_chunking_params.py     # Chunking 参数实验（8种配置）
│   ├── test_priority2_questions.py # 低相似度问题测试
│   ├── test_query_rewrite.py       # Query Rewrite 实验
│   └── test_improvements.py        # 改进效果对比
│
├── 数据
│   ├── benchmark.json              # 23个评测问题
│   ├── source_documents/Le.pdf     # 生物塑料论文
│   └── 实验报告.md                  # 完整实验记录
│
└── 配置
    ├── requirements.txt
    ├── .env.example                # 环境变量模板
    └── INSTALL.md
```

## 🚀 快速开始

### 1. 安装依赖

```bash
python -m venv rag-env
source rag-env/bin/activate  # Windows: rag-env\Scripts\activate
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入你的 API Key
```

`.env` 文件内容：
```
DEEPSEEK_API_KEY=your_api_key_here
CHROMA_PERSIST_DIR=doc_index
```

### 3. 构建向量索引

```bash
# 原始配置
python index_documents.py

# 或使用优化配置 (overlap=150)
python index_documents_improved.py
```

### 4. 运行问答

```bash
# 批量评测
python batch_qa.py

# 交互式 UI
streamlit run document_chatbot_ui.py
```

### 5. 运行实验

```bash
# Chunking 参数优化
python test_chunking_params.py

# Query Rewrite 实验
python test_query_rewrite.py
```

## 🔬 核心发现

### 1. 相似度计算修正

**问题**：Chroma 的 `similarity_search_with_score()` 返回欧几里得距离，非余弦相似度

**解决方案**：手动计算余弦相似度

$$\cos(\theta) = \frac{q \cdot d}{|q| \cdot |d|}$$

### 2. Chunking 优化的局限性

- ✅ 对 "跨边界信息" 问题有效（如 ID 5, 18）
- ❌ 对 "术语不匹配" 问题无效（如 ID 4, 7, 16, 20, 23）

### 3. Query Rewrite 是最有效的改进

| 方法 | 适用场景 | 成功率 | 平均提升 |
|------|---------|--------|---------|
| Chunking 优化 | 跨边界信息 | 20% | +11.4% |
| **Query Rewrite** | **术语不匹配** | **100%** | **+45.4%** |

## 📈 后续优化方向

- [ ] **Embedding 模型升级**：all-MiniLM-L6-v2 → all-mpnet-base-v2
- [ ] **混合检索**：向量检索 + BM25 关键词检索
- [ ] **重排序 (Reranking)**：LLM 对检索结果重打分
- [ ] **Answerability 检测**：识别不可回答问题，避免幻觉

## 📄 评测标准

- **相似度**：余弦相似度，范围 [0, 1]
- **检索质量**：HIGH (≥0.55) / LOW (<0.55)
- **答案正确性**：精确匹配 / 包含匹配 / 数值匹配

## 📚 参考

- [LangChain Documentation](https://python.langchain.com/)
- [Chroma Vector Database](https://www.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)

## 📝 License

MIT License
