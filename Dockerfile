FROM python:3.10-slim

# 작업 디렉토리 생성
WORKDIR /app

# 로컬 소스 복사
COPY . .

# pip 업그레이드 및 의존성 설치
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# 포트 오픈 (Streamlit용)
EXPOSE 8501

# 실행 스크립트 실행
CMD ["bash", "run_all.sh"]
