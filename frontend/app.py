import streamlit as st
import random
import time
from streamlit_chat import message


# 페이지 설정
st.set_page_config(
    page_title="RAG 챗봇",
    page_icon="💬",
    initial_sidebar_state="expanded"
)

# 제목과 소개
st.title('환영합니다! 👋')
st.markdown('## 박물관 챗봇')

st.write(""" 박물관 견학 중 모르시는게 있으시면 말씀해주세요. 해당 부분에 대해서 자세하게 설명드리겠습니다.""")

# 사이드바 
st.sidebar.header('채팅 내역')
option = st.sidebar.selectbox(
    '원하는 기능을 선택하세요:',
    ['데이터 보기', '차트 보기', '정보']
)

# 채팅 메시지 함수 (좌우 말풍선 구현)
def chat_message(message, is_user=True):
    align = "flex-end" if is_user else "flex-start"
    bubble_color = "#dcf8c6" if is_user else "#ffffff"
    avatar = "🧑" if is_user else "🤖"

    st.markdown(f"""
    <div style="display: flex; justify-content: {align}; margin-bottom: 10px;">
        <div style="max-width: 70%; background-color: {bubble_color}; 
                    padding: 10px 15px; border-radius: 15px; font-size: 16px;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
            <b>{avatar}</b> {message}
        </div>
    </div>
    """, unsafe_allow_html=True)

# user_input = st.text_input("메시지를 입력하세요", key="user_input")


input_container = st.container()
with input_container:
    st.markdown('<div class="chat-input-container">', unsafe_allow_html=True)
    user_input = st.text_input("메시지를 입력하세요", key="user_input", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("보내기"):
        chat_message(user_input, is_user=True)
        chat_message("안녕하세요! 무엇을 도와드릴까요?", is_user=False)

    if user_input:
        # st.session_state.chat_history.append(("user", user_input))
        # st.session_state.chat_history.append(("bot", "안녕하세요! 무엇을 도와드릴까요?"))
        # st.experimental_rerun()
        chat_message(user_input, is_user=True)
        chat_message("안녕하세요! 무엇을 도와드릴까요?", is_user=False)
    
