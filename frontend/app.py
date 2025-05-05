import sys
import os
import requests
import streamlit as st
import time

# 상위 디렉토리의 rag_pipeline 사용 시 경로 추가 (현재는 FastAPI 호출로 미사용)
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# from rag.rag_pipeline import run_rag_pipeline

# FastAPI 백엔드에 POST 요청을 보내는 함수
def query_backend_rag(question: str) -> str:
    try:
        response = requests.post(
            "http://localhost:8000/rag",
            json={"question": question}
        )
        return response.json().get("answer", "[❗️답변을 불러오는 데 실패했습니다]")
    except Exception as e:
        return f"[❗️에러 발생: {e}]"

# 페이지 설정
st.set_page_config(
    page_title="RAG 챗봇",
    page_icon="💬",
    initial_sidebar_state="expanded"
)

# 타이틀
st.title('환영합니다! 👋')
st.markdown('## 국립중앙도서관')
st.write("모르시는 게 있으시면 말씀해주세요. 해당 부분에 대해서 자세하게 설명드리겠습니다.")

# 사이드바
st.sidebar.header('채팅 내역')
option = st.sidebar.selectbox(
    '원하는 기능을 선택하세요:',
    ['데이터 보기', '차트 보기', '정보']
)

# 세션에 대화 기록 저장
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # {"role": "user"/"assistant", "content": str} 리스트

# 채팅 말풍선 렌더링 함수
def chat_message(msg: str, is_user: bool = True):
    align = "flex-end" if is_user else "flex-start"
    bubble_color = "#dcf8c6" if is_user else "#ffffff"
    avatar = "🧑" if is_user else "🤖"

    st.markdown(f"""
    <div style="display: flex; justify-content: {align}; margin-bottom: 10px;">
        <div style="max-width: 70%; background-color: {bubble_color}; 
                    padding: 10px 15px; border-radius: 15px; font-size: 16px;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
            <b>{avatar}</b> {msg}
        </div>
    </div>
    """, unsafe_allow_html=True)

# 사용자 입력란
user_input = st.text_input("메시지를 입력하세요", label_visibility="collapsed")

# 전송 버튼
if st.button("보내기") and user_input:
    # 1) 사용자 메시지 저장 및 출력
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    chat_message(user_input, is_user=True)

    # 2) FastAPI RAG API 호출
    with st.spinner("답변 생성 중..."):
        answer = query_backend_rag(user_input)
        time.sleep(0.3)  # UX용 딜레이

    # 3) 봇 메시지 저장 및 출력
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    chat_message(answer, is_user=False)

    # 4) 입력란 초기화
    st.session_state.user_input = ""

# 이전 대화 기록 렌더링
for chat in st.session_state.chat_history:
    if chat["role"] == "user":
        chat_message(chat["content"], is_user=True)
    else:
        chat_message(chat["content"], is_user=False)
