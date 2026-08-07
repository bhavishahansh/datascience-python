from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

from app.rag.prompt import RAG_PROMPT


def retrieve_node(state, retriever):

    last_question = ""

    for message in reversed(state["messages"]):

        if isinstance(message, HumanMessage):
            last_question = message.content
            break

    docs = retriever.invoke(last_question)

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    return {
        "context": context
    }


def generate_node(state, llm):

    chain = (
        RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    answer = chain.invoke(
        {
            "context": state["context"],
            "messages": state["messages"]
        }
    )

    return {

        "messages": [
            AIMessage(content=answer)
        ]

    }