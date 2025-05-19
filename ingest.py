import os
import logging
import warnings
from datetime import datetime
import time
import torch
from PyPDF2 import PdfReader
from langchain.text_splitter import TokenTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import DocArrayHnswSearch

# ——— Settings ———
MANUALS_FOLDER = r"C:\Users\akram\OneDrive\Documents\Manuals"
CHUNK_SIZE     = 500
CHUNK_OVERLAP  = 50
EMBED_MODEL    = "all-MiniLM-L6-v2"
WORK_DIR       = "hnsw_manuals"

# ——— Logging setup ———
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Silence DocArray internals
logging.getLogger("docarray").setLevel(logging.WARNING)

# (Optional) Silence that Pydantic migration warning
warnings.filterwarnings(
    "ignore",
    message=r"`pydantic\.error_wrappers:ValidationError`"
)

def ingest_folder(folder_path: str):
    start_time = datetime.now()
    t0 = time.perf_counter()

    processed_files = skipped_files = pages_processed = 0
    docs = []

    splitter = TokenTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        model_name="gpt2"
    )

    # 1) Read & chunk PDFs
    for root, _, files in os.walk(folder_path):
        for fn in files:
            if not fn.lower().endswith(".pdf"):
                continue
            path = os.path.join(root, fn)
            try:
                with open(path, "rb") as fh:
                    reader = PdfReader(fh)
                    processed_files += 1
                    for page_num, page in enumerate(reader.pages, start=1):
                        pages_processed += 1
                        text = page.extract_text() or ""
                        for chunk in splitter.split_text(text):
                            docs.append(
                                Document(
                                    page_content=chunk,
                                    metadata={"source": path, "page": page_num}
                                )
                            )
            except Exception as e:
                skipped_files += 1
                logging.warning(f"❌ Skipping {path!r}: {e}")

    t1 = time.perf_counter()
    logging.info(f"[+] Files processed: {processed_files}, skipped: {skipped_files}")
    logging.info(f"[+] Pages processed: {pages_processed}")
    logging.info(f"[+] Chunks created: {len(docs)}")
    logging.info(f"[+] Parsing & chunking took {t1 - t0:.2f}s")

    # 2) Device info
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    logging.info(f"[+] Using device: {device}")

    # 3) Embedding + detect dimension
    logging.info(f"[+] Embedding with model `{EMBED_MODEL}`…")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    if not docs:
        raise RuntimeError("No documents to embed!")

    sample_vec = embeddings.embed_query(docs[0].page_content)
    n_dim = len(sample_vec)
    logging.info(f"[+] Detected embedding dimension: {n_dim}")

    # 4) Build & persist HNSW index
    t2 = time.perf_counter()
    vectorstore = DocArrayHnswSearch.from_documents(
        documents=docs,
        embedding=embeddings,
        work_dir=WORK_DIR,
        n_dim=n_dim
    )
    t3 = time.perf_counter()
    logging.info(f"[+] Embedding + indexing took {t3 - t2:.2f}s")

    # 5) Finish timing
    end_time = datetime.now()
    total = (end_time - start_time).total_seconds()
    logging.info(f"[+] Ingestion started at {start_time}")
    logging.info(f"[+] Ingestion finished at {end_time}")
    logging.info(f"[+] Total elapsed time: {total:.2f}s")

if __name__ == "__main__":
    ingest_folder(MANUALS_FOLDER)
