from uuid import uuid4

import streamlit as st
import torch

import sys
import os

# HuggingFace_RAGFlow/ 경로를 sys.path에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.chain_creator import ConversationalChain

torch.classes.__path__ = []


def main():
    st.title("Chatbot Assistant")

    # Initialize session state for chat history and model
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "model_loaded" not in st.session_state:
        st.session_state.model_loaded = False

    if not st.session_state.model_loaded:
        # Load the model only once
        with st.spinner("Loading model..."):
            st.session_state.chatbot_model = ConversationalChain().chain_with_history(
                str(uuid4())
            )
            st.session_state.model_loaded = True

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Get user input
    user_input = st.chat_input("Type your message here...")
    # 임시 코드
    print(f"사용자 입력:{user_input}")

    if user_input:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Display user message
        with st.chat_message("user"):
            st.write(user_input)

        # Generate response
        with st.chat_message("assistant"), st.spinner("Thinking..."):
            response = st.session_state.chatbot_model.invoke({"question": user_input})
            print(f"LLM응답: {response}")
            st.write(response)

        # Add assistant response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
