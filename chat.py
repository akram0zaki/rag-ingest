import os
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import DocArrayHnswSearch
from langchain_community.llms import LlamaCpp

# ——— Settings ———
WORK_DIR    = "hnsw_manuals"
EMBED_MODEL = "all-MiniLM-L6-v2"
MODEL_PATH  = "C:/Users/akram/OneDrive/Documents/Workspace/AI/llama.cpp/llama-3.1-8B.gguf"
CTX_SIZE    = 2048
TOP_K       = 4

def setup():
    # 1) Embeddings + detect dimension
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    sample_vec = embeddings.embed_query("test")
    n_dim = len(sample_vec)

    # 2) Load (or create) the HNSW index from work_dir
    vectorstore = DocArrayHnswSearch.from_params(
        embedding=embeddings,
        work_dir=WORK_DIR,
        n_dim=n_dim
    )

    # 3) Load local LLaMA model
    llm = LlamaCpp(model_path=MODEL_PATH, n_ctx=CTX_SIZE)
    return vectorstore, llm

def answer_query(query: str, vectorstore, llm) -> str:
    docs = vectorstore.similarity_search(query, k=TOP_K)

    parts = []
    for doc in docs:
        src  = os.path.basename(doc.metadata.get("source", "")) or "unknown.pdf"
        page = doc.metadata.get("page", "?")
        txt  = doc.page_content.strip()
        parts.append(f"[Source: {src}, page {page}]\n{txt}")

    context = "\n\n---\n\n".join(parts)
    prompt = f"""You are an expert technical assistant. Use the following excerpts—each tagged with its source filename and page—to answer the question.

Context:
{context}

Question: {query}
Answer:"""

    return llm(prompt).strip()

def main():
    vs, llm = setup()
    print("🛠️  Manual Assistant ready! (Ctrl+C to exit.)")
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
