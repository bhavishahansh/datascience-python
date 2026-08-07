import langchain
print(langchain.__version__)

try:
    from langchain.memory import ConversationBufferWindowMemory
    print("Import successful from langchain.memory!")
except ImportError:
    try:
        from langchain.memory.buffer_window import ConversationBufferWindowMemory
        print("Import successful from langchain.memory.buffer_window!")
    except ImportError as e:
        print("Error details:", e)