from langchain_core.output_parsers import StrOutputParser

from app.rag.prompt import RAG_PROMPT


class GeneratorNode:

    def __init__(self, llm):

        self.llm = llm

    def __call__(self, state):

        chain = (

            RAG_PROMPT

            | self.llm

            | StrOutputParser()

        )

        answer = chain.invoke(

            {

                "context": state["context"],

                "question": state["question"]

            }

        )

        state["answer"] = answer

        return state