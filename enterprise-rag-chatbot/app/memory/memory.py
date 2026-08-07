from app.memory.state import ChatState


def update_chat_history(state: ChatState):

    history = state.get("chat_history", [])

    history.append(
        {
            "question": state["question"],
            "answer": state["answer"]
        }
    )

    state["chat_history"] = history

    return state