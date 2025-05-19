# ingest.py

import os
import logging
from PyPDF2 import PdfReader
from langchain.text_splitter import TokenTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
from langchain.vectorstores import HNSWLib

# ——— Settings ———
MANUALS_FOLDER = r"C:\Users\akram\OneDrive\Documents\Manuals"
CHUNK_SIZE     = 500
CHUNK_OVERLAP  = 50
EMBED_MODEL    = "all-MiniLM-L6-v2"
OUTPUT_DIR     = "faiss_manuals"

logging.basicConfig(level=logging.INFO)

def ingest_folder(folder_path: str):
    texts, metadatas = [], []
    splitter = TokenTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        model_name="gpt2"
    )

    # 1. Walk through all PDFs recursively
    for root, _, files in os.walk(folder_path):
        for fn in files:
            if not fn.lower().endswith(".pdf"):
                continue
            path = os.path.join(root, fn)
            try:
                with open(path, "rb") as fh:
                    reader = PdfReader(fh)
                    for page_num, page in enumerate(reader.pages, start=1):
                        text = page.extract_text() or ""
                        for chunk in splitter.split_text(text):
                            texts.append(chunk)
                            metadatas.append({
                                "source": path,
                                "page": page_num
                            })
            except Exception as e:
                logging.warning(f"❌ Skipping {path!r}: {e}")

    logging.info(f"[+] Collected {len(texts)} chunks")

    # 2. Init HuggingFaceEmbeddings (wraps sentence-transformers under the hood)
    logging.info(f"[+] Embedding chunks with `{EMBED_MODEL}`")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    # 3. Build & save FAISS index with metadata
    logging.info(f"[+] Building FAISS index and saving to `{OUTPUT_DIR}/`")
    vectorstore = HNSWLib.from_texts(
        texts,
        embeddings,
        metadatas=metadatas
    )
    vectorstore.save_local(OUTPUT_DIR)
    logging.info("[+] Ingestion complete.")

if __name__ == "__main__":
    ingest_folder(MANUALS_FOLDER)
