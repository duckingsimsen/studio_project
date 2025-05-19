import asyncio

from langchain_faiss import FAISS
from langchain_community.vectorstores import FAISS
from core.config import LLMConfig
from core.model import Embedding

class VectorStoreRetriever:
    def __init__(self):
        # 저장된 FAISS 벡터 DB 불러오기
        self.embedding_model = Embedding.load_embeddings(LLMConfig.MODEL_NAME)
        self.vector_store = FAISS.load_local(
            folder_path=LLMConfig.FAISS_DB_STORE_PATH,
            embeddings=self.embedding_model,
        )

    def retrieve_similar_documents(self, query: str, top_k: int = 5): # 여기서 top-k의 갯수를 정할 수 있음
        """
        사용자 입력(query)을 임베딩하고,
        FAISS Vector Store에서 비슷한 문서 top_k개를 검색해서 반환한다.
        """
        try:
            docs_and_scores = self.vector_store.similarity_search_with_score(query, k=top_k)
            return docs_and_scores  # [(Document, 점수)] 형태
        except Exception as ex:
            raise Exception(f"Error retrieving documents from FAISS: {ex}") from ex

if __name__ == "__main__":
    vector_store = VectorStoreRetriever()
