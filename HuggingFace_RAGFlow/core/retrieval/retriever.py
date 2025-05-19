import logging
import numpy as np

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
            folder_path=LLMConfig.FAISS_DB_STORE_PATH,
            embeddings=self._embeddings,
            allow_dangerous_deserialization=True,
        )

        # 🔽 index_to_docstore_id의 key, value를 문자열로 변환
        self._vector_store.index_to_docstore_id = {
            int(idx): str(doc_id) for idx, doc_id in self._vector_store.index_to_docstore_id.items()
        }

        # 🔽 docstore 내부 key들도 문자열로 강제 변환
        if hasattr(self._vector_store.docstore, "_dict"):
            self._vector_store.docstore._dict = {
                str(k): v for k, v in self._vector_store.docstore._dict.items()
            }

        test_vector = self._embeddings.embed_query("무엇을 도와드릴까요?")
        print("✅ FAISS에 들어가는 벡터 dtype:", np.array(test_vector).dtype)

    def retrieve(self):
        try:
            return self._vector_store.as_retriever()
        except Exception as ex:
            logging.error("Error by getting retriever")
            raise ex
    
    def create_retriever(self, db:FAISS):
        return db.as_retriever(
            search_kwargs = {
                "k": 5,
                "fetch_k": 15,
                "mmr": True,
                "mmr_beta": 0.3
            }
        )