
# Setup

cd C:\Users\akram\OneDrive\Documents\Workspace\AI

mkdir rag-ingest

cd rag-ingest

mkdir models

python -m venv .venv

.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt

OR
pip install langchain
pip install llama-cpp-python        # for local Llama-style models
pip install sentence-transformers   # for embeddings
pip install faiss-cpu               # vector store
pip install PyPDF2                  # PDF text extraction
pip install tiktoken                # chunking on token bounds
pip install gradio                  # simple UI

** Note: If you prefer an alternative to FAISS, you can swap in Chroma or Weaviate.

# Prerequisites to llama.cpp
1. Install libcurl (via vcpkg)
- Clone & bootstrap vcpkg:
git clone https://github.com/microsoft/vcpkg.git C:\Users\akram\OneDrive\Apps\vcpkg
cd C:\Users\akram\OneDrive\Apps\vcpkg
.\bootstrap-vcpkg.bat

2. Install curl for x64:
cd C:\Users\akram\OneDrive\Apps\vcpkg
.\vcpkg.exe install curl:x64-windows

# Model
1. Go to https://huggingface.co/datasets/meta-llama/Llama-3.1-8B-Instruct-evals
Accept the license to download the model. You should receive an email with download link if request is approved.

2. Tell CMake where vcpkg lives (once per machine)
set VCPKG_ROOT=C:\Users\akram\OneDrive\Apps\vcpkg
set CMAKE_TOOLCHAIN_FILE=%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake

3. Clone and build llama.cpp
cd C:\Users\akram\OneDrive\Documents\Workspace\AI
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
mkdir build
cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE="C:/Users/akram/OneDrive/Apps/vcpkg/scripts/buildsystems/vcpkg.cmake" -DVCPKG_TARGET_TRIPLET="x64-windows" -DCURL_INCLUDE_DIR="C:/Users/akram/OneDrive/Apps/vcpkg/installed/x64-windows/include" -DCURL_LIBRARY="C:/Users/akram/OneDrive/Apps/vcpkg/installed/x64-windows/lib/libcurl.lib"

cmake --build . --config Release

4. Download the model
- Install LFS
git lfs install

- Clone the llama3.1 repo
cd C:\Users\akram\OneDrive\Documents\Workspace\AI

Clone the repo (prompts for a HuggingFace Access Token):
git clone https://huggingface.co/meta-llama/Llama-3.1-8B

(or download the files manually from https://huggingface.co/meta-llama/Llama-3.1-8B/tree/main)

cd Llama-3.1-8B

Authenticate to Hugging Face (Make sure hugging face cli is installed pip install -U "huggingface_hub[cli]"):
huggingface-cli login

Fetch the actual weight files via LFS:
git lfs pull

5. Convert to GGUF (or legacy GGML) format (run from inside llama.cpp repo):
python .\convert_hf_to_gguf.py `
  C:\Users\akram\OneDrive\Documents\Workspace\AI\Llama-3.1-8B `
  --outfile llama-3.1-8B.gguf

6. (OR) Download a pre-converted model:
git lfs install
git clone https://huggingface.co/TheBloke/Llama-2-7B-chat-GGUF

7. Point llama-cpp-python at the new model. In chat.py, set:
MODEL_PATH = "C:/Users/akram/OneDrive/Documents/Workspace/AI/llama.cpp/llama-3.1-8B.gguf"  # or .bin, whichever you produced
llm = LlamaCpp(model_path=MODEL_PATH, n_ctx=2048)

