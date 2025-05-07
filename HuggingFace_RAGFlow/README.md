# HuggingFace RAGFlow
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
## Overview
This project implements a classic Retrieval-Augmented Generation (RAG) system using HuggingFace models with quantization techniques. The system processes PDF documents, extracts their content, and enables interactive question-answering through a Streamlit web application.

## Prerequisites
- [Anaconda](https://www.anaconda.com/download/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed on your system
- Python 3.12 or higher

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/edcalderin/HuggingFace_RAGFlow.git
cd HuggingFace_RAGFlow
```

### 2. Create and activate the Conda environment
```bash
# Create a new Conda environment
conda env create -n hg_ragflow --file requirements.txt

# Activate the environment
conda activate hg_ragflow
```

On Windows, you might need to use:
```bash
source activate hg_ragflow
```

If you have GPU
```
pip3 install torch --index-url https://download.pytorch.org/whl/cu126
```

### 3. Verify the installation
```bash
# Verify that the environment is active
conda info --envs

# The active environment should be marked with an asterisk (*)
```

## Usage

### Development workflow
1. Rename `.env.example` to `.env` and set the `HUGGINGFACE_TOKEN` variable with your own HuggingFace token https://huggingface.co/settings/tokens

2. Load embeddings to Qdrant Vector Store:
   ```bash
   python -m core.data_loader.vector_store
   ```

3. Run Streamlit app:
    ```bash
    python -m streamlit run app/streamlit.py
    ```

### Configuration
Located `core/config.py` and feel free to edit these global parameters:

```python
@dataclass(frozen=True)
class LLMConfig:
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-mpnet-base-v2" <-- embedding model
    COLLECTION_NAME: str = "historiacard_docs"
    QDRANT_STORE_PATH: str = "./tmp" <-- directory to Qdrant vector store

    # Model
    MODEL_NAME: str = "meta-llama/Llama-3.2-3B-Instruct"
    MODEL_TASK: str = "text-generation" <-- task type
    TEMPERATURE: float = 0.1
    MAX_NEW_TOKENS: int = 1024
```
### Lint
Style the code with Ruff:

```bash
ruff format .
ruff check . --fix
```
### Deactivating the environment
When you're done working on the project, deactivate the Conda environment:

```bash
conda deactivate
```

**Last but not least:**  
Locate you cache directory and remove embedding and model directory used by the project, as these may occupy several gigabytes of storage.

## Environment Configuration

### Requirements
The project includes an `requirements.txt` file that defines all required dependencies. Here's what it looks like:

```bash
accelerate==1.5.2
bitsandbytes==0.45.3
langchain-community==0.3.19
langchain-core==0.3.44
langchain-huggingface==0.1.2
langchain-qdrant==0.2.0
pypdf==5.3.1
python-dotenv==1.0.1
ruff==0.9.10
streamlit==1.43.2
torch==2.6.0+cu126
transformers==4.49.0
```

## Project Structure
```
HuggingFace_RAGFlow/
├── app/                   # Streamlit app
│   ├── streamlit.py       # Main application entry point
├── core/                  # LLM stuff
│   ├── chain_creator/     # Files to create conversational chain and memory management
│   └── data_loader/       # Files to save embeddings to Vector Store.
│   └── model/             # LLM Model and Embeddings
│   └── retrieval/         # Vector Store Retriever
│   └── utils/             # Logging configuration
│   └── config.py          # Global configuration parameters
└── README.md              # This file
```

## Contact
**LinkedIn:** https://www.linkedin.com/in/erick-calderin-5bb6963b/  
**e-mail:** edcm.erick@gmail.com

Just in case, feel free to create an issue 😊

In .env directory, you need to input your Huggingface token.
When use need token model