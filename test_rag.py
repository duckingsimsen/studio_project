# test_rag_loop.py

from rag.rag_pipeline import run_rag_pipeline

def main():
    print("📌 PDF 기반 RAG 질의응답 테스트 (종료하려면 'exit' 입력)")
    while True:
        question = input("\n❓ 질문: ")
        if question.lower() in ["exit", "quit"]:
            print("🛑 종료합니다.")
            break
        answer = run_rag_pipeline(question)
        print("💬 답변:", answer)

if __name__ == "__main__":
    main()
