import streamlit as st
import tiktoken
from loguru import logger

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import UnstructuredPowerPointLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS

# from streamlit_chat import message
from langchain.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory

def main():
    st.set_page_config(page_title="DirChat", page_icon=":books:")  # 상단 탭 꾸미는곳

    st.title("_Private Data :red[QA Chat]_ :books:")

    if "conversation" not in st.session_state: # conversation 초기화 하는 부분
        st.session_state.conversation = None

    if "chat_history" not in st.session_state: 
        st.session_state.chat_history = None

    # with st.sidebar: dl 사이드바 추가하는 부분

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role" : "assistant", "content" : "안녕하세요! 주어진 문서에 대해 궁금하신 것이 있으면 언제든 물어봐주세요!"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # history = StreamlitChatMessageHistory(key = "chat_messages")

    # Chat logic
    if query := st.chat_input("질문을 입력해주세요."):
        st.session_state.messages.append({"role" : "user", "content" : query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"): # 질문한 부분을 session_state에 올리는 부분
            chain = st.session_state.conversation

            with st.spinner("Thinking..."): # 챗봇이 답을 하는 부분
                result = chain({"question": query})
                with get_openai_callback() as cb:
                    st.session_state.chat_history = result['chat_history']
                response = result['answer']
                source_documents = result['source_documents']

                st.markdown(response)
                with st.expander("참고 문서 확인"):
                    st.markdown(source_documents[0].metadata['source'], help = source_documents[0].page_content)
                    st.markdown(source_documents[1].metadata['source'], help = source_documents[1].page_content)
                    st.markdown(source_documents[2].metadata['source'], help = source_documents[2].page_content)
                    


# Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
                                        model_name = "jhgan/ko-sroberta-multitask",
                                        model_kwargs = {'device': 'cpu'},
                                        # model_kwargs = "auto"
                                        encode_kwargs = {'normalize_embeddings': True}
                                        )  
    vectordb = FAISS.from_documents(text_chunks, embeddings)
    return vectordb

def get_conversation_chain(vetorestore,openai_api_key):
    llm = ChatOpenAI(openai_api_key = openai_api_key, model_name = 'gpt-3.5-turbo', temperature = 0)
    conversation_chain = ConversationalRetrievalChain.from_llm(
            llm = llm, 
            chain_type = "stuff", 
            retriever=vetorestore.as_retriever(search_type = 'mmr', vervose = True), 
            memory = ConversationBufferMemory(memory_key = 'chat_history', return_messages = True, output_key = 'answer'),
            get_chat_history = lambda h: h,
            return_source_documents = True,
            verbose = True
        )

    return conversation_chain

if __name__ == '__main__':
    main()