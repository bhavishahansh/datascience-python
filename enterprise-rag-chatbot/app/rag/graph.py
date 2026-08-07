from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from app.rag.state import GraphState
from app.rag.node import retrieve_node
from app.rag.node import generate_node

def build_graph(retriever, llm):

    workflow = StateGraph(GraphState)

    workflow.add_node(
        "retrieve",
        lambda state: retrieve_node(
            state,
            retriever,
            llm
        )
    )

    workflow.add_node(
        "generate",
        lambda state: generate_node(
            state,
            llm
        )
    )

    workflow.add_edge(
        START,
        "retrieve"
    )

    workflow.add_edge(
        "retrieve",
        "generate"
    )

    workflow.add_edge(
        "generate",
        END
    )

    memory = MemorySaver()

    return workflow.compile(
        checkpointer=memory
    )