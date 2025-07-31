import gradio as gr

from chatbot import on_chat
from corpus import CORPUS, on_upload, n_tokens
from config import OLLAMA_MODEL, OPENAI_API_KEY

def build_ui():
    with gr.Blocks(title="RALF: Retrieval Agent for Long Files") as demo:
        gr.Markdown("## 📄 → 🤖  RALF: Retrieval Agent for Long Files")

        with gr.Row():
            uploader = gr.File(label="Upload PDF / DOCX / TXT", file_count="multiple")
            status   = gr.Markdown()

        chatbot = gr.Chatbot(value=[], height=400)

        with gr.Row():
            mode = gr.Radio(
                ["Question-Answering", "Retrieval"],
                value="Question-Answering", label="Mode"
            )
            with gr.Column():
                llm_provider = gr.Dropdown(
                    ["OpenAI", "Ollama"], value="OpenAI", label="LLM Provider"
                )
                openai_api_key = gr.Textbox(
                    label="OpenAI API Key", type="password", interactive=True, visible=True
                )
                ollama_model = gr.Textbox(
                    value=OLLAMA_MODEL, label="Ollama Model", interactive=True, visible=False
                )
        
        llm_provider.change(
            lambda value: gr.update(visible=value == "OpenAI"),
            inputs=llm_provider,
            outputs=openai_api_key,
            queue=False,
        ).then(
            lambda value: gr.update(visible=value == "Ollama"),
            inputs=llm_provider,
            outputs=ollama_model,
            queue=False,
        )

        uploader.change(on_upload, inputs=uploader, outputs=[status, chatbot])

        txt  = gr.Textbox(lines=2, placeholder="Ask…", autofocus=True)
        send = gr.Button("Send")
        status_message = gr.Markdown("")

        send.click(
            on_chat,
            inputs=[txt, chatbot, mode, llm_provider, ollama_model, openai_api_key],
            outputs=[chatbot, txt, send, status_message]
        )
        txt.submit(
            on_chat,
            inputs=[txt, chatbot, mode, llm_provider, ollama_model, openai_api_key],
            outputs=[chatbot, txt, send, status_message]
        )

        gr.Markdown("Made with ❤️  &  OpenAI — only API calls leave your machine.")
    return demo
