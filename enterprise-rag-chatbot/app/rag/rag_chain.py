from langchain_core.output_parsers import StrOutputParser


from app.rag.prompt import RAG_PROMPT


class RAGChain:

    def __init__(self, retriever, llm):

        self.retriever = retriever
        self.llm = llm
        self.chat_history = []

    def ask(self, question):

        docs = self.retriever.invoke(question)

        context = "\n\n".join(
            doc.page_content
            for doc in docs
        )

        chain = (
            RAG_PROMPT  | self.llm  | StrOutputParser()
        )

        
        answer = chain.invoke(
            {
                "context": context,
                "question": question,
                "chat_history":self.chat_history
            }
        )

        self.chat_history.append(
        {
            "question": question,
            "answer": answer
        }
        )

        return answer