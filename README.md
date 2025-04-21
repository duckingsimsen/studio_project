# LangChain PDF-based RAG with FAISS

## Overview

이 프로젝트는 PDF 문서를 기반으로 Retrieval‑Augmented Generation(RAG) 파이프라인을 구축하고,
FAISS 벡터 데이터베이스를 통해 효율적인 검색 기능을 제공하며, Docker 컨테이너로 배포하는 것을 목표로 합니다.

## Features

- PDF 문서 파싱 및 추출
- LangChain을 이용한 텍스트 분할(chunking) 및 임베딩 생성
- FAISS를 사용한 벡터 데이터베이스 구축 및 유사도 검색
- 사용자의 질문에 답하는 RAG 서비스
- Docker화된 컨테이너로 간편 배포
