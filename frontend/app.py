import streamlit as st
import time
from rag_pipeline import run_rag_pipeline  # rag 폴더 안의 rag_pipeline.py

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

    # 2) RAG 파이프라인 호출
    with st.spinner("답변 생성 중..."):
        answer = run_rag_pipeline(user_input)
        time.sleep(0.3)  # UX용 딜레이

    # 3) 봇 메시지 저장 및 출력
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    chat_message(answer, is_user=False)

    # 4) 입력란 초기화
    st.session_state.user_input = ""

# 이전 대화 기록 렌더링
for chat in st.session_state.chat_history:
    # 이미 화면에 출력된 메시지도 다시 출력되지만,
    # 스크롤을 유지하기 위해 반복 렌더링합니다.
    if chat["role"] == "user":
        chat_message(chat["content"], is_user=True)
    else:
        chat_message(chat["content"], is_user=False)
