import os
import gradio as gr
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import DocArrayHnswSearch
from langchain_community.llms import LlamaCpp

# ——— Settings ———
WORK_DIR    = "hnsw_manuals"
EMBED_MODEL = "all-MiniLM-L6-v2"
MODEL_PATH  = r"C:\Users\akram\OneDrive\Documents\Workspace\AI\llama.cpp\llama-3.1-8B.gguf"
CTX_SIZE    = 2048
TOP_K       = 4

# Prepare these once globally
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
# detect embedding dim
_n_dim = len(embeddings.embed_query("test"))
llm = LlamaCpp(model_path=MODEL_PATH, n_ctx=CTX_SIZE)

def answer_query(query: str, vectorstore, llm, k: int = TOP_K) -> str:
    docs = vectorstore.similarity_search(query, k=k)
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

def chat_fn(user_input):
    # Re-create the DocArrayHnswSearch index **inside** this thread
    vectorstore = DocArrayHnswSearch.from_params(
        embedding=embeddings,
        work_dir=WORK_DIR,
        n_dim=_n_dim,
    )
    return answer_query(user_input, vectorstore, llm, k=TOP_K)

iface = gr.Interface(
    fn=chat_fn,
    inputs="text",
    outputs="text",
    title="Home Device Assistant",
    description="Ask questions about your PDF manuals."
)

if __name__ == "__main__":
    iface.launch()
