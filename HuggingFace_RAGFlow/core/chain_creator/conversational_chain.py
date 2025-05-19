import logging
from operator import itemgetter
from pathlib import Path

from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory

from core.model import LLMModel
from core.retrieval import Retriever
from core.utils.logging import setup_logging
# from core.qa_store.qa_storeage import SaveQA

setup_logging()


class ConversationalChain:
    def __init__(self) -> None:
        self._llm_model: LLMModel = LLMModel()
        self._retriever = Retriever()
        self._llm = self._llm_model.create_pipeline()
        # self.qa_saver = SaveQA()
        self._store: dict = {}

    def _get_prompt_template(self) -> ChatPromptTemplate:
        """
        The function reads a chat prompt template from a file and returns it as a
        ChatPromptTemplate object.

        Returns:
          The function `_get_prompt_template` is returning an instance of
          `ChatPromptTemplate` created from the content read from the file
          "prompt.txt".
        """
        try:
            prompt_path: Path = Path(__file__).parent / "prompt.txt"
            prompt: str = prompt_path.read_text(encoding = "utf-8-sig")
            return ChatPromptTemplate.from_template(prompt)
        except FileNotFoundError as ex:
            raise ex

    def _get_chain(self) -> RunnableSequence:
        def fetch_context(input: dict) -> dict:
            question = input["question"]
            docs = self._retriever.retrieve().get_relevant_documents(question)

            if not docs or all(len(doc.page_content.strip()) == 0 for doc in docs):
                # context가 없을 경우 LLM에게 전달할 메시지를 고정
                return {
                    "question": question,
                    "context": "죄송합니다. 해당 질문에 대한 정보가 현재 문서에 없습니다."
                }

            context = "\n\n".join([doc.page_content for doc in docs])
            return {
                "question": question,
                "context": context
            }

        return (
            RunnableLambda(fetch_context)
            | self._get_prompt_template()
            | self._llm
            | StrOutputParser()
        )


    def _get_memory_by_session_id(self, session_id: str) -> BaseChatMessageHistory:
        """
        This function retrieves a chat message history object based on a session ID,
        creating a new one if it doesn't already exist.

        Args:
          session_id: The `session_id` parameter is a string that represents the unique
          identifier for a chat session. It is used to retrieve the chat message
          history associated with that particular session from the `_store` dictionary.
          If the chat message history for the given `session_id` does not already exist
          in the `_store

        Returns:
          The chat message history associated with the provided session ID. If the
          session ID is not found in the store, a new InMemoryChatMessageHistory object
          is created and stored in the store before being returned.
        """
        logging.info(f"Reading history messages from session_id: {session_id}")
        if session_id not in self._store:
            self._store[session_id] = InMemoryChatMessageHistory()
        return self._store[session_id]

    def chain_with_history(self, session_id: str) -> RunnableWithMessageHistory:
        """
        The function `chain_with_history` returns a `RunnableWithMessageHistory` object
        with specified parameters.

        Args:
            session_id: Uuid used as `session_id`
        Returns:
          An instance of the `RunnableWithMessageHistory` class is being returned.
        """
        return RunnableWithMessageHistory(
            self._get_chain(),
            self._get_memory_by_session_id,
            input_messages_key="question",
            history_messages_key="history",
        ).with_config(config={"session_id": session_id})

    # def run_chain(self, session_id: str, question: str) -> str:
    #     chain = self.chain_with_history(session_id)
    #     answer = chain.invoke({"question": question})

    #     self.qa_saver.save(question, answer)

    #     return answer
    
    # def maybe_use_cached_anwer(self, question: str, threshold: float = 0.2):
    #     results = self.qa_saver.search(question, top_k = 1)

    #     if results:
    #         doc, score = results[0]
    #         if score < threshold:
    #             return doc.page_content
            
    #     return None