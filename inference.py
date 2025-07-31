# ---------------------------------------------------------------------------
# Environment & configuration
# ---------------------------------------------------------------------------

API_KEY = ""
OPENAI_MODEL = "gpt-4o"
MAX_DOC_TOKENS= 10000000
client = OpenAI(api_key=API_KEY)

# ──────────────────  Token counting helper  ──────────────────
try:
    import tiktoken  # type: ignore
    tok_enc = tiktoken.encoding_for_model(OPENAI_MODEL)
    n_tokens = lambda text: len(tok_enc.encode(text))           # noqa: E731
except ImportError:
    n_tokens = lambda text: max(1, len(text.split()) // 0.75)   # noqa: E731
# ─────────────────── File-parsing helpers ───────────────────────────────────
def read_pdf(fh: io.BufferedReader) -> str:
    import pypdf
    return "\n".join(p.extract_text() or "" for p in pypdf.PdfReader(fh).pages)

def read_docx(fh: io.BufferedReader) -> str:
    from docx import Document
    return "\n".join(p.text for p in Document(fh).paragraphs)

_HANDLERS = {".pdf": read_pdf, ".docx": read_docx}

# ─────────────────── Corpus store ───────────────────────────────────────────
class Corpus:
    def __init__(self):
        self.files: List[str] = []
        self.raw = ""

    def clear(self):
        self.files.clear()
        self.raw = ""

    def add(self, file: gr.File):
        ext = Path(file.name).suffix.lower()
        with open(file.name, "rb") as fh:
            txt = _HANDLERS.get(ext, lambda h: h.read().decode(errors="ignore"))(fh)
        self.files.append(Path(file.name).name)
        self.raw += "\n" + txt
        while n_tokens(self.raw) > MAX_DOC_TOKENS:
            self.raw = self.raw[int(len(self.raw) * .1):]

CORPUS = Corpus()

# ─────────────────── System prompts ─────────────────────────────────────────
QA_SYS = (
    "You are a helpful assistant. Use ONLY the information in the documents "
    "below. If the answer is not present, say you don't know.\n\n---\n{docs}\n---"
)
RET_SYS = (
    "You are a helpful assistant performing pure retrieval. "
    "Return the smallest possible quote(s) that answer the user query. "
    "If nothing matches, reply exactly: 'Information not found in corpus.'. "
    "Preface each quote with the filename in brackets if available.\n\n---\n{docs}\n---"
)

def ask_llm(prompt: str, history: list[dict], mode: str) -> str:
    system = QA_SYS if mode == "Question-Answering" else RET_SYS
    msgs = [{"role": "system", "content": system.format(docs=CORPUS.raw[:16000])}] \
           + history + [{"role": "user", "content": prompt}]
    rsp = client.chat.completions.create(
        model=OPENAI_MODEL, messages=msgs,
        temperature=0 if mode == "Retrieval" else 0.2,
        max_tokens=1024,
    )
    return rsp.choices[0].message.content.strip()

# ─────────────────── Highlight helper ───────────────────────────────────────
def highlight_quotes(raw_answer: str) -> str:
    """Return answer with context & <mark>highlight</mark> for each quote line."""
    if raw_answer.strip() == "Information not found in corpus.":
        return raw_answer

    snippets: list[str] = []
    corpus_lower = CORPUS.raw.lower()

    for line in raw_answer.splitlines():
        txt = line.strip()
        if not txt:
            continue

        # Extract optional [filename] prefix
        file_tag, quote = None, txt
        if txt.startswith("["):
            end = txt.find("]")
            if end != -1:
                file_tag = txt[1:end]
                quote = txt[end + 1:].strip()

        # Remove leading punctuation / quotes the model might add
        quote_clean = quote.strip(" “”\"'")
        if not quote_clean:
            continue

        # Locate quote in corpus (case-insensitive)
        idx = corpus_lower.find(quote_clean.lower())
        if idx == -1:
            # fallback: couldn’t find, keep model line
            snippets.append(txt)
            continue

        start = max(0, idx - 120)
        end   = min(len(CORPUS.raw), idx + len(quote_clean) + 120)
        context = (
            CORPUS.raw[start:idx]
            + "<mark>" + CORPUS.raw[idx:idx + len(quote_clean)] + "</mark>"
            + CORPUS.raw[idx + len(quote_clean):end]
        )
        prefix = f"[{file_tag}] " if file_tag else ""
        snippets.append(prefix + "…" + context.strip() + "…")

    return "\n\n".join(snippets) if snippets else raw_answer

# ─────────────────── Callbacks ──────────────────────────────────────────────
def on_upload(files: list[gr.File]):
    CORPUS.clear()
    missing_docx = False
    for f in files or []:
        try: CORPUS.add(f)
        except ModuleNotFoundError as e:
            if "docx" in str(e).lower(): missing_docx = True

    names = ("\n• " + "\n• ".join(CORPUS.files)) if CORPUS.files else ""
    warn  = "\n⚠️ Install **python-docx** for DOCX extraction." if missing_docx else ""
    status = (
        f"Loaded {len(CORPUS.files)} file(s):{names}\n"
        f"Tokens stored: {n_tokens(CORPUS.raw)}.{warn}"
    )
    return status, []

def on_chat(msg: str, history: list | None, mode: str):
    history = history or []
    raw_answer = ask_llm(msg, history, mode)
    answer = highlight_quotes(raw_answer) if mode == "Retrieval" else raw_answer
    history += [{"role": "user", "content": msg},
                {"role": "assistant", "content": answer}]
    return history

# ─────────────────── UI ─────────────────────────────────────────────────────
def build_ui():
    with gr.Blocks(title="Local Doc-Chat (OpenAI)") as demo:
        gr.Markdown("## 📄 → 🤖  Local Doc-Chat & Retrieval with Highlight")

        with gr.Row():
            uploader = gr.File(label="Upload PDF / DOCX / TXT", file_count="multiple")
            status   = gr.Markdown()

        chatbot = gr.Chatbot(value=[], type="messages", height=400)

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

# ─────────────────── Main ───────────────────────────────────────────────────
if __name__ == "__main__":
    build_ui().launch()