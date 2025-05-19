# rag-ingest

## Overview

**rag-ingest** is a simple pipeline for ingestion of documents into a vector store and serving them for retrieval-augmented generation (RAG). I am using this project to ingest the user manuals of all devices in my house and then chat with Llama 3.1 for troubleshooting any issues I have. I created this repo for educational purpose only, CPUs are not as efficient as GPUs when it comes to inference so for a real use-case it makes sense to work with CUDA cores for siginifcant performance gains.

This project allows you to:

- Prepare and convert language models to the **gguf** format.
- Split and embed text documents using CPU-agnostic libraries.
- Store embeddings in a vector store for fast similarity search.

## Goals

1. **CPU-Agnostic**: Use libraries that run efficiently on any CPU without requiring GPU acceleration.
2. **Modular Pipeline**: Separate steps for model preparation, document ingestion, and query serving.
3. **Reproducible**: Clear instructions to download models, convert formats, and ingest data.

## Requirements

- Python 3.8+
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) (for gguf model loading)
- [sentence-transformers](https://www.sbert.net/) or [transformers](https://github.com/huggingface/transformers) (CPU mode)
- [faiss-cpu](https://github.com/facebookresearch/faiss) or [chromadb](https://github.com/chroma-core/chroma)
- Hugging Face CLI (`huggingface_hub`)

## Model Preparation

The below instructions are generic so you can download the model of your choosing. I went for Llama 3.1 8B where I downloaded the model files then converted it to gguf format. I tracked the instructions for my own setup in SETUP.md.

1. **Get a pre-trained model**  
   ```bash
   pip install huggingface_hub
   huggingface-cli login
   huggingface-cli repo clone <model-id> ./model
   ```

2. **Convert to GGUF**  
   GGUF is the llama.cpp “general-purpose” file format for quantized inference.  
   ```bash
   git clone https://github.com/ggerganov/llama.cpp
   cd llama.cpp
   # build the converter
   make
   # convert
   ./convert-ggml-to-gguf models/<model>.bin models/<model>.gguf
   ```

## What is GGUF?

**GGUF** is a binary format optimized for the [llama.cpp](https://github.com/ggerganov/llama.cpp) runtime. It supports quantized weights, fast loading, and is CPU-friendly.

## Document Ingestion & Vector Store

1. **Text Splitting**  
   Use a library like `langchain` or `nltk` to split large documents into smaller chunks.

2. **Embedding**  
   Compute embeddings with `sentence-transformers` or `transformers` in CPU mode:
   ```python
   from transformers import AutoTokenizer, AutoModel
   tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
   model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
   ```

3. **Vector Store**  
   Store embeddings using FAISS or Chroma:
   ```python
   import faiss
   index = faiss.IndexFlatL2(embeddings.shape[1])
   index.add(embeddings)
   ```

## What is a Vector Store?

A **vector store** is a database of numeric embeddings that supports fast similarity search. It allows you to retrieve the most relevant document chunks given a query embedding.

## Pipeline Flow

```mermaid
flowchart TB
  subgraph Model Preparation
    A[Download HF Model] --> B[Convert to GGUF]
    B --> C[Load GGUF Model]
  end
  subgraph Ingestion
    D[Raw Documents] --> E[Text Splitting]
    E --> F[Compute Embeddings]
    F --> G[Store Embeddings in Vector Store]
  end
  subgraph Query
    H[User Query] --> I[Embed Query]
    I --> J[Vector Store Lookup]
    J --> K[Retrieve Chunks]
    K --> L[LLM Inference (gguf)]
    L --> M[Answer]
  end
```

## Usage

1. Clone this repo:
   ```bash
   git clone https://github.com/akram0zaki/rag-ingest.git
   cd rag-ingest
   ```
2. Follow **Model Preparation** and **Document Ingestion** steps above.
3. Run your query script pointing at the vector store and gguf model.

---
