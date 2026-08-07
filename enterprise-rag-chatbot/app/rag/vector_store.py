from langchain_community.vectorstores import FAISS


def save_vector_store(chunks, embedding_model):

    vector_db = FAISS.from_documents(
        chunks,
        embedding_model
    )

    vector_db.save_local("data/vectorstore")

    return vector_db


def load_vector_store(embedding_model):

    return FAISS.load_local(
        "data/vectorstore",
        embedding_model,
        allow_dangerous_deserialization=True
    )