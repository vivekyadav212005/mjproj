from rag.llm_client import GeminiClient

client = GeminiClient()
print(client.chat([{"role": "user", "content": "Say hello"}]))