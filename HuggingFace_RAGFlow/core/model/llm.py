import torch
from langchain_huggingface import HuggingFacePipeline
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)
from transformers.utils.logging import set_verbosity_error

from core.config import LLMConfig

set_verbosity_error()


class LLMModel:
    # def __init__(self) -> None:
    #     self._bnb_config: BitsAndBytesConfig = BitsAndBytesConfig(
    #         load_in_4bit=True,
    #         bnb_4bit_quant_type="nf4",
    #         bnb_4bit_compute_dtype=torch.bfloat16,
    #         bnb_4bit_use_double_quant=False,
    #     )

    def _load_model(self) -> AutoModelForCausalLM:
        """
        The function `_load_model` loads a model for causal language modeling using
        Hugging Face's `AutoModelForCausalLM` with quantization settings.

        Returns:
        The `_load_model` function returns an instance of `AutoModelForCausalLM`
        loaded from the Hugging Face model with the specified configuration settings.
        """
        try:
            return AutoModelForCausalLM.from_pretrained(
                LLMConfig.MODEL_NAME,
                device_map="auto",
                # quantization_config=self._bnb_config,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                trust_remote_code = True
            )
        except Exception as ex:
            raise Exception(f"Error loading model from HF: {ex}") from ex

    def _load_tokenizer(self):
        """
        The function `_load_tokenizer` attempts to load a tokenizer using Hugging
        Face's `AutoTokenizer` with error handling.

        Returns:
          The `_load_tokenizer` method returns the tokenizer that is loaded using the
        `AutoTokenizer.from_pretrained` method.
        """
        try:
            tokenizer = AutoTokenizer.from_pretrained(LLMConfig.MODEL_NAME)
            tokenizer.pad_token = tokenizer.eos_token
            return tokenizer
        except Exception as ex:
            raise Exception(f"Error loading tokenizer from HF: {ex}") from ex

    def create_pipeline(self) -> HuggingFacePipeline:
        """
        The function `_create_pipeline` creates a Hugging Face pipeline for text
        generation using a given model.

        Returns:
        An instance of the `HuggingFacePipeline` class with the text generation
        pipeline created using the provided model and configuration parameters.
        """
        try:
            pipe: pipeline = pipeline(
                model=self._load_model(),
                tokenizer=self._load_tokenizer(),
                task=LLMConfig.MODEL_TASK,
                do_sample=True,
                repetition_penalty=1.1,
                return_full_text=False,
                max_new_tokens=LLMConfig.MAX_NEW_TOKENS,
                temperature=LLMConfig.TEMPERATURE,
                trust_remote_code = True
            )
            return HuggingFacePipeline(pipeline=pipe)
        except Exception as ex:
            raise Exception(f"Error creating HF Pipeline: {ex}") from ex
