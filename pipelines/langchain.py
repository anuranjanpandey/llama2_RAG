from langchain.llms import HuggingFacePipeline
from steps import base_llm

def generate_text(text: str):
    base_llm = base_llm.BaseLLM()  # Define the missing "base_llm" variable
    llm = HuggingFacePipeline(pipeline=base_llm)

    print(llm(prompt="Explain to me the difference between nuclear fission and fusion."))
