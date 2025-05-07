import logging
import os

import torch
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


# class Embedding:
#     @staticmethod
#     def load_embeddings(model_name: str) -> HuggingFaceEmbeddings:
#         """
#         The function `_load_embeddings` returns a `HuggingFaceEmbeddings` object with
#         a specified model name and location.

#         Returns:
#         An instance of the `HuggingFaceEmbeddings` class.
#         """
#         try:
#             device: torch.device = torch.device(
#                 "cuda" if torch.cuda.is_available() else "cpu"
#             )
#             logging.info(f"Running on {device}")
#             if device == "cuda":
#                 logging.info(f"Device: {torch.cuda.get_device_name(0)}")

#             return HuggingFaceEmbeddings(
#                 model_name=model_name,
#                 multi_process=True,
#                 model_kwargs={"device": "cpu"},
#                 encode_kwargs={"normalize_embeddings": True},
#             )
#         except Exception as ex:
#             raise Exception(f"Error loading HuggingFace embeddings: {ex}") from ex

class Embedding:
    @staticmethod
    def load_embeddings(model_name: str) -> HuggingFaceEmbeddings:
        """
        Load HuggingFace embedding model and force it to run on CPU.
        """
        try:
            logging.info("Forcing embedding model to run on CPU")

            return HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": "cpu"},           # ⬅️ 반드시 CPU
                encode_kwargs={"normalize_embeddings": True},
                multi_process=False                        # ⬅️ CUDA 경합 방지 위해 멀티 프로세스 끔
            )
        except Exception as ex:
            raise Exception(f"Error loading HuggingFace embeddings: {ex}") from ex