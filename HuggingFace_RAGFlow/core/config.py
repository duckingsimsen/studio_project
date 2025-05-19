from dataclasses import dataclass


@dataclass(frozen=True)
class LLMConfig:
    EMBEDDING_MODEL_NAME: str = "nlpai-lab/KURE-v1"
    COLLECTION_NAME: str = "historiacard_docs"
    FAISS_DB_STORE_PATH: str = "./tmp/faiss_db_fixed"

    # Model
    # MODEL_NAME: str = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
    MODEL_NAME: str = "naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B"
    MODEL_TASK: str = "text-generation"
    TEMPERATURE: float = 0.7
    MAX_NEW_TOKENS: int = 1024
