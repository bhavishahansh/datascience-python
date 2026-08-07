from langchain_ollama import ChatOllama


class LLMFactory:

    @staticmethod
    def get_llm(provider):

        if provider == "ollama":

            return ChatOllama(
                model="llama3.2",
                temperature=0
            )

        raise ValueError(f"Unsupported provider: {provider}")