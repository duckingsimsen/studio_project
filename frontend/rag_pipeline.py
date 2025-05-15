import os
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline as REMOVEDpipeline
)

def embed_query(text: str) -> list[float]:
    """텍스트를 임베딩 벡터로 변환"""
    vec = embed_model.encode(text, convert_to_numpy=True)
    return vec.astype(np.float32).tolist()

CACHE_DIR = os.path.join(os.path.dirname(__file__), "model_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

embed_model = SentenceTransformer("nlpai-lab/KURE-v1", cache_folder=CACHE_DIR)

faiss_db = FAISS.load_local("faiss_db_0416", embed_query, allow_dangerous_deserialization = True)

# index_to_docstore_id 키를 문자열로 변환
faiss_db.index_to_docstore_id = {
    idx: str(doc_id)
    for idx, doc_id in faiss_db.index_to_docstore_id.items()
}

# docstore 내부 _dict 키도 문자열로 변환
if hasattr(faiss_db.docstore, '_dict'):
    faiss_db.docstore._dict = {
        str(k): v
        for k, v in faiss_db.docstore._dict.items()
    }
retriever = faiss_db.as_retriever(search_kwargs={"k": 5})


exaone_model_id = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
# quant_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype="bfloat16"
# )

tokenizer = AutoTokenizer.from_pretrained(
    exaone_model_id,
    cache_dir=CACHE_DIR,
    use_fast=True
)

model = AutoModelForCausalLM.from_pretrained(
    exaone_model_id,
    cache_dir=CACHE_DIR,
    # quantization_config=quant_config,
    device_map="auto",
    trust_remote_code=True
)

textgen_pipeline = REMOVEDpipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
    max_new_tokens=256,
    do_sample=True,
    temperature=0.7
)
llm = HuggingFacePipeline(pipeline=textgen_pipeline)


rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)

def run_rag_pipeline(question: str) -> str:
    """
    주어진 질문에 대해
    1) FAISS DB에서 관련 문서 검색
    2) 양자화된 LLM으로 최종 답변 생성
    """
    return rag_chain.run(question)