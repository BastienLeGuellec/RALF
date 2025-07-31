import gradio as gr

from chatbot import on_chat
from corpus import CORPUS, on_upload, n_tokens

def build_ui():
    with gr.Blocks(title="RALF: Retrieval Agent for Long Files") as demo:
        gr.Markdown("## 📄 → 🤖  RALF: Retrieval Agent for Long Files")

        with gr.Row():
            uploader = gr.File(label="Upload PDF / DOCX / TXT", file_count="multiple")
            status   = gr.Markdown()

        chatbot = gr.Chatbot(value=[], height=400)

        mode = gr.Radio(
            ["Question-Answering", "Retrieval"],
            value="Question-Answering", label="Mode"
        )

        uploader.change(on_upload, inputs=uploader, outputs=[status, chatbot])

        txt  = gr.Textbox(lines=2, placeholder="Ask…", autofocus=True)
        send = gr.Button("Send")
        send.click(on_chat, inputs=[txt, chatbot, mode], outputs=chatbot)
        txt.submit(on_chat, inputs=[txt, chatbot, mode], outputs=chatbot)

        gr.Markdown("Made with ❤️  &  OpenAI — only API calls leave your machine.")
    return demo
