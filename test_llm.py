# test_llm.py

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="qwen3-4b-instruct-2507",
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    temperature=0
)

response = llm.invoke(
    "What is LangChain?"
)

print(response.content)