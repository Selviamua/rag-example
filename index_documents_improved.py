"""改进版文档索引 - 增加 overlap 和优化分割策略
实验目标：通过增加 chunk overlap 改善检索质量
"""

import os
from dotenv import load_dotenv

load_dotenv()

from transformers import AutoTokenizer
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

SOURCE_DOCUMENTS = ["source_documents/Le.pdf"]
COLLECTION_NAME = "doc_index_v2"  # 使用新的 collection 名称
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# 改进参数
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50  # 增加 overlap（原来是 0）


def main():
    print("="*80)
    print("【改进版文档索引】")
    print("="*80)
    print(f"Chunk Size: {CHUNK_SIZE}")
    print(f"Chunk Overlap: {CHUNK_OVERLAP} (原: 0)")
    print(f"Collection Name: {COLLECTION_NAME}")
    print()
    
    print("正在加载文档...")
    all_docs = ingest_docs(SOURCE_DOCUMENTS)
    print(f"✓ 共切分为 {len(all_docs)} 个文档块")
    
    print("\n正在生成向量索引...")
    db = generate_embed_index(all_docs)
    print("✓ 索引生成完成")
    
    print("\n" + "="*80)
    print("完成！新索引已保存到 collection:", COLLECTION_NAME)
    print("="*80)


def ingest_docs(source_documents):
    all_docs = []
    for source_doc in source_documents:
        print(f"  处理: {source_doc}")
        docs = pdf_to_chunks(source_doc)
        all_docs = all_docs + docs
    return all_docs


def pdf_to_chunks(pdf_file):
    """使用改进的分割策略"""
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    
    # 改进的分割器：增加 overlap
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        separators=["\n\n", "\n", ". ", " ", ""],  # 优化分隔符顺序
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    
    loader = PyPDFLoader(pdf_file)
    docs = loader.load_and_split(text_splitter)
    return docs


def generate_embed_index(docs):
    """生成向量索引"""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    chroma_persist_dir = os.getenv("CHROMA_PERSIST_DIR")
    
    if not chroma_persist_dir:
        raise EnvironmentError("未找到 CHROMA_PERSIST_DIR 环境变量")
    
    db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=chroma_persist_dir,
    )
    db.persist()
    return db


if __name__ == "__main__":
    main()
