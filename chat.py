# chat.py

import os
from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
from langchain.vectorstores import HNSWLib
from langchain_community.llms import LlamaCpp

# ——— Settings ———
FAISS_DIR   = "faiss_manuals"
EMBED_MODEL = "all-MiniLM-L6-v2"
MODEL_PATH  = "C:/Users/akram/OneDrive/Documents/Workspace/AI/llama.cpp/llama-3.1-8B.gguf"
CTX_SIZE    = 2048
TOP_K       = 4

def setup():
    # 1. Use the community embeddings (implements embed_documents, etc.)
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    # 2. Load FAISS index with explicit allow_dangerous_deserialization
    vectorstore = HNSWLib.load_local(
        FAISS_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )

    # 3. Use the community LLM interface
    llm = LlamaCpp(model_path=MODEL_PATH, n_ctx=CTX_SIZE)

    return vectorstore, llm

def answer_query(query: str, vectorstore, llm, k: int = TOP_K) -> str:
    docs_and_scores = vectorstore.similarity_search_with_score(query, k=k)
    context_parts = []
    for doc, score in docs_and_scores:
        src  = os.path.basename(doc.metadata.get("source", "")) or "unknown.pdf"
        page = doc.metadata.get("page", "?")
        txt  = doc.page_content.strip()
        context_parts.append(f"[Source: {src}, page {page}]\n{txt}")

    context = "\n\n---\n\n".join(context_parts)
    prompt = f"""You are an expert technical assistant. Use the following device manual excerpts—each preceded by its source filename and page number—to answer the question.

Context:
{context}

Question: {query}
Answer:"""

    return llm(prompt).strip()

def main():
    vs, llm = setup()
    print("🛠️  Manual Assistant ready. (Ctrl+C to exit.)")
    try:
        while True:
            q = input("\nAsk: ").strip()
            if not q:
                continue
            print("\n" + answer_query(q, vs, llm) + "\n")
    except KeyboardInterrupt:
        print("\nGoodbye!")

if __name__ == "__main__":
    main()
