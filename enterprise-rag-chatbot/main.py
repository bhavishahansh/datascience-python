from langchain_core.messages import HumanMessage

from app.rag.pdf_loader import load_pdf
from app.rag.chunking import split_documents
from app.rag.embedding import load_embedding_model
from app.rag.vector_store import load_vector_store
from app.rag.graph import build_graph
from app.rag.llm import LLMFactory


print("Loading PDF...")

documents = load_pdf(
    "data/documents/Employee-Handbook.pdf"
)

print(f"Loaded {len(documents)} pages")

chunks = split_documents(documents)

print(f"Chunks : {len(chunks)}")

embedding_model = load_embedding_model()

vector_db = load_vector_store(
    
    embedding_model
)

retriever = vector_db.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10
    }
)

llm = LLMFactory.get_llm("ollama")

graph = build_graph(
    retriever,
    llm
)

thread_id = "employee_chat"

while True:

    question = input("\nYou : ")

    if question.lower() == "exit":
        break

    result = graph.invoke(

        {
            "messages": [
                HumanMessage(content=question)
            ]
        },

        config={
            "configurable": {
                "thread_id": "employee_chat"
            }
        }

    )

    print("\nAssistant :\n")

    print(result["messages"][-1].content)