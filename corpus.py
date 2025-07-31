import gradio as gr
import io
from pathlib import Path
from typing import List

from utils import n_tokens

def read_pdf(fh: io.BufferedReader) -> str:
    import pypdf
    return "\n".join(p.extract_text() or "" for p in pypdf.PdfReader(fh).pages)

def read_docx(fh: io.BufferedReader) -> str:
    from docx import Document
    return "\n".join(p.text for p in Document(fh).paragraphs)

_HANDLERS = {".pdf": read_pdf, ".docx": read_docx}

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

CORPUS = Corpus()

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