import os
from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import HuggingFaceEndpoint


repo_id="mistralai/Mistral-7B-Instruct-v0.3"
llm=HuggingFaceEndpoint(repo_id=repo_id,max_length=150,temperature=0.7,token=os.getenv("HF_TOKEN"))
print(llm)

print(llm.invoke("What is machine learning"))
print(llm.invoke("What is generative AI "))

repo_id="google/gemma-2-9b"
llm=HuggingFaceEndpoint(repo_id=repo_id,max_length=150,temperature=0.7,token=os.getenv("HF_TOKEN"))
print(llm)