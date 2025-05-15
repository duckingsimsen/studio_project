import streamlit as st
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.rag_pipeline import run_rag_pipeline
from langchain.memory import StreamlitChatMessageHistory

def main():
    st.set_page_config(page_title="DirChat", page_icon=":books:")
    st.title("_Private Data :red[QA Chat]_ :books:")

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [
            {"role": "assistant", "content": "안녕하세요! 문서 기반 질문에 대해 도와드릴게요. 궁금한 내용을 입력해주세요!"}
        ]

    # 이전 메시지 출력
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    history = StreamlitChatMessageHistory(key="chat_messages")

    # 사용자 질문 입력
    if query := st.chat_input("질문을 입력해주세요."):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = run_rag_pipeline(query)
                except Exception as e:
                    response = f"❗ 오류가 발생했습니다: {e}"
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
