from llm import ask_llm, highlight_quotes

import gradio as gr
from llm import ask_llm, highlight_quotes

def on_chat(msg: str, history: list | None, mode: str, llm_provider: str, ollama_model: str, openai_api_key: str):
    history = history or []
    llm_history = []
    for user_msg, assistant_msg in history:
        llm_history.append({"role": "user", "content": user_msg})
        if assistant_msg is not None:
            llm_history.append({"role": "assistant", "content": assistant_msg})

    # Disable input and show thinking message
    yield history, gr.update(value="", interactive=False), gr.update(interactive=False), "Thinking..."

    raw_answer = ask_llm(msg, llm_history, mode, llm_provider, ollama_model, openai_api_key)
    answer = highlight_quotes(raw_answer) if mode == "Retrieval" else raw_answer
    history.append((msg, answer))

    # Re-enable input and clear thinking message
    yield history, gr.update(value="", interactive=True), gr.update(interactive=True), ""
