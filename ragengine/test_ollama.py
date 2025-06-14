from langchain_community.chat_models import ChatOllama

llm = ChatOllama(model="llama3")

response = llm.invoke("Hello, who are you?")
print("🔁 Bot says:", response)
