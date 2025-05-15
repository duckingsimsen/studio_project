from fastapi import FastAPI
from pydantic import BaseModel
from frontend.rag_pipeline import run_rag_pipeline  # 같은 루트에 rag 폴더 있다고 가정

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/rag")
async def rag_answer(query: Query):
    answer = run_rag_pipeline(query.question)
    return {"answer": answer}
