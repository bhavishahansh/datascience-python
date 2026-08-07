class RetrieverNode:

    def __init__(self, retriever):
        self.retriever = retriever

    def __call__(self, state):

        docs = self.retriever.invoke(state["question"])

        context = "\n\n".join(
            doc.page_content
            for doc in docs
        )

        state["context"] = context

        return state