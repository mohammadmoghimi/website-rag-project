from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="gemma2:2b"
)

response = llm.invoke("can you help me with homework ?")

print(response.content)