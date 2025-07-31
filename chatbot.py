from llm import ask_llm, highlight_quotes

def on_chat(msg: str, history: list | None, mode: str):
    history = history or []
    llm_history = []
    for user_msg, assistant_msg in history:
        llm_history.append({"role": "user", "content": user_msg})
        if assistant_msg is not None:
            llm_history.append({"role": "assistant", "content": assistant_msg})

    raw_answer = ask_llm(msg, llm_history, mode)
    answer = highlight_quotes(raw_answer) if mode == "Retrieval" else raw_answer
    history.append((msg, answer))
    return history
