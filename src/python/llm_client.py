import httpx
from langchain_groq import ChatGroq

class LLMClient:
    def __init__(self):
        self.llm = ChatGroq(temperature=0.0, 
                            groq_api_key="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",
                            http_client=httpx.Client(verify=False),model="deepseek-r1-distill-llama-70b"
                            )

