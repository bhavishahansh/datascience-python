from typing import TypedDict


class ChatState(TypedDict):
    question: str
    context: str
    answer: str
    chat_history: list