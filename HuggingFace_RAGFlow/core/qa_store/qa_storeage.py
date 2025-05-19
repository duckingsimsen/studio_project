import os
from langchain.vectorstores import FAISS
from langchain.schema import Document
from core.model import Embedding
from core.config import LLMConfig

class SaveQA:
    def __init__(self, faiss_path: str = "qa_faiss_store"):
        self.faiss_path = faiss_path
        self.embedding_model = Embedding.load_embeddings(LLMConfig.EMBEDDING_MODEL_NAME)

        if os.path.exists(self.faiss_path):
            self.vectorstore = FAISS.load_local(
                self.faiss_path,
                embeddings=self.embedding_model,
                allow_dangerous_deserialization=True
            )
        else:
            self.vectorstore = None

    def save(self, question: str, answer: str):
        doc = Document(page_content=answer, metadata={"question": question})
        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents([doc], embedding=self.embedding_model)
        else:
            self.vectorstore.add_documents([doc])
        self.vectorstore.save_local(self.faiss_path)
        print(f"✅ 저장 완료: {question}")

    def search(self, query: str, top_k: int = 3):
        if self.vectorstore is None:
            return []
        return self.vectorstore.similarity_search_with_score(query, k=top_k)