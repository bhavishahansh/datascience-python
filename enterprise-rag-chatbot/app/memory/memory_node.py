from app.memory.memory import update_chat_history


class MemoryNode:

    def __call__(self, state):

        update_chat_history(

            state["question"],

            state["answer"]

        )

        return state