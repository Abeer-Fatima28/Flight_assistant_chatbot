import os
from langchain.memory import ConversationBufferWindowMemory

class ConversationMemory:
    def __init__(self, k: int | None = None):
        if k is None:
            try:
                k = int(os.getenv("MEMORY_K", "8"))
            except Exception:
                k = 8
        self.mem = ConversationBufferWindowMemory(k=k, return_messages=False, memory_key="history")
    
    def add_user(self, text: str):
        self.mem.chat_memory.add_user_message(text)
    
    def add_ai(self, text: str):
        self.mem.chat_memory.add_ai_message(text)
    
    def get_formatted(self) -> str:
        return self.mem.load_memory_variables({}).get("history", "")
