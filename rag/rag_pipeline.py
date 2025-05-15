import os
import yaml
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline
)

logging.basicConfig(level=logging.INFO)
BASE_DIR = os.path.dirname(__file__)
CACHE_DIR = os.path.join(BASE_DIR, "model_cache")
PROMPT_PATH = os.path.join(BASE_DIR, "prompt.yaml")
FAISS_DIR = "faiss_db_0416"
os.makedirs(CACHE_DIR, exist_ok=True)

def load_prompt_template(path: str) -> PromptTemplate:
    with open(path, "r", encoding="utf-8") as f:
        prompt_config = yaml.safe_load(f)
    return PromptTemplate(
        template=prompt_config["prompt_template"],
        input_variables=["context", "question"]
    )

def load_embedding_model(model_name: str = "nlpai-lab/KURE-v1") -> SentenceTransformer:
    return SentenceTransformer(model_name, cache_folder=CACHE_DIR)

def embed_query(text: str, model: SentenceTransformer) -> list[float]:
    vec = model.encode(text, convert_to_numpy=True)
    return vec.astype(np.float32).tolist()

def load_faiss_db(embed_fn) -> FAISS:
    db = FAISS.load_local(FAISS_DIR, embed_fn, allow_dangerous_deserialization=True)
    db.index_to_docstore_id = {idx: str(doc_id) for idx, doc_id in db.index_to_docstore_id.items()}
    if hasattr(db.docstore, "_dict"):
        db.docstore._dict = {str(k): v for k, v in db.docstore._dict.items()}
    return db

def create_retriever(db: FAISS):
    return db.as_retriever(
        search_kwargs={
            "k": 5,
            "fetch_k": 15,
            "mmr": True,
            "mmr_beta": 0.3
        }
    )

def load_quantized_llm(model_id: str) -> HuggingFacePipeline:
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="bfloat16"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=CACHE_DIR, use_fast=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=CACHE_DIR,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=quant_config
    )
    textgen_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
        max_new_tokens=1024,
        temperature=0.7,
        top_k=50,
        repetition_penalty=1.03
    )
    return HuggingFacePipeline(pipeline=textgen_pipe)

def create_rag_chain(llm, retriever, prompt: PromptTemplate) -> RetrievalQA:
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )

prompt = load_prompt_template(PROMPT_PATH)
embedding_model = load_embedding_model()
embed_fn = lambda x: embed_query(x, embedding_model)
faiss_db = load_faiss_db(embed_fn)
retriever = create_retriever(faiss_db)
llm = load_quantized_llm("LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct")
rag_chain = create_rag_chain(llm, retriever, prompt)

def run_rag_pipeline(question: str) -> str:
    try:
        logging.info(f"Processing question: {question}")
        return rag_chain.run(question)
    except Exception as e:
        logging.error(f"Failed to generate answer: {str(e)}")
        return "답변을 생성하는 데 실패했습니다."


