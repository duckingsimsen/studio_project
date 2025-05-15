import logging

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from core.config import LLMConfig
from core.model import Embedding
from core.utils.logging import setup_logging

setup_logging()


class Retriever:
    def __init__(self) -> None:
        self._embeddings: HuggingFaceEmbeddings = Embedding.load_embeddings(
            model_name=LLMConfig.EMBEDDING_MODEL_NAME
        )
        logging.info("Loading FAISS retriever")
        self._vector_store = FAISS.load_local(
            folder_path=LLMConfig.QDRANT_STORE_PATH,
            embeddings=self._embeddings,
            allow_dangerous_deserialization=True
        )

    def retrieve(self):
        try:
            return self._vector_store.as_retriever()
        except Exception as ex:
            logging.error("Error by getting retriever")
            raise ex