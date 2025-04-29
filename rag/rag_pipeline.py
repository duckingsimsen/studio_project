from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

repo_id = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"

llm = HuggingFaceEndpoint(repo_id = repo_id, max_length = 32768, temperature = 0.1)
