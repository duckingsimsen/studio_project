import logging
import os
import numpy as np

from dotenv import load_dotenv
from huggingface_hub import login
from langchain_huggingface import HuggingFaceEmbeddings

from core.utils.logging import setup_logging

load_dotenv()

try:
    login(token=os.getenv("HUGGINGFACE_TOKEN"))
except Exception as ex:
    logging.error(f"Error logging in to Hugging Face: {ex}")

setup_logging()

class Embedding:
    @staticmethod
    def load_embeddings(model_name: str) -> HuggingFaceEmbeddings:
        """
        Load HuggingFace embedding model and force it to run on CPU.
        """
        try:
            logging.info("Forcing embedding model to run on CPU")
            
            model = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={
                    "normalize_embeddings": True,
                    # "dtype": "float32"
                    },
                multi_process=False
            )

            test_vector = model.embed_query("테스트 문장입니다.")
            logging.info(f"✅ 테스트 벡터 dtype 확인: {np.array(test_vector).dtype}")

            return model

        except Exception as ex:
            raise Exception(f"Error loading HuggingFace embeddings: {ex}") from ex
