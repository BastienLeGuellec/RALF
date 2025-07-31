from openai import OpenAI

from config import OPENAI_MODEL, OPENAI_API_KEY
from corpus import CORPUS

client = OpenAI(api_key=OPENAI_API_KEY)

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
    msgs = [{"role": "system", "content": system.format(docs=CORPUS.raw)}] \
           + history + [{"role": "user", "content": prompt}]
    rsp = client.chat.completions.create(
        model=OPENAI_MODEL, messages=msgs,
        temperature=0 if mode == "Retrieval" else 0.2,
        max_tokens=1024,
    )
    return rsp.choices[0].message.content.strip()

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
