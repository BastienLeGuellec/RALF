from openai import OpenAI
import requests
import json

from config import OLLAMA_HOST, OLLAMA_MODEL
from corpus import CORPUS


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

def ask_llm(prompt: str, history: list[dict], mode: str, llm_provider: str, ollama_model: str, openai_api_key: str) -> str:
    if llm_provider == "OpenAI":
        return ask_openai(prompt, history, mode, openai_api_key)
    elif llm_provider == "Ollama":
        return ask_ollama(prompt, history, mode, ollama_model)
    else:
        raise ValueError(f"Unknown LLM provider: {llm_provider}")

def ask_openai(prompt: str, history: list[dict], mode: str, openai_api_key: str) -> str:
    if not openai_api_key:
        raise ValueError("OpenAI API key not configured in UI.")
    client = OpenAI(api_key=openai_api_key)
    system = QA_SYS if mode == "Question-Answering" else RET_SYS
    msgs = [{"role": "system", "content": system.format(docs=CORPUS.raw)}] \
           + history + [{"role": "user", "content": prompt}]
    rsp = client.chat.completions.create(
        model="gpt-4o", messages=msgs,
        temperature=0 if mode == "Retrieval" else 0.2,
        max_tokens=1024,
    )
    return rsp.choices[0].message.content.strip()

def ask_ollama(prompt: str, history: list[dict], mode: str, ollama_model: str) -> str:
    system = QA_SYS if mode == "Question-Answering" else RET_SYS
    msgs = [{"role": "system", "content": system.format(docs=CORPUS.raw)}] \
           + history + [{"role": "user", "content": prompt}]
    
    try:
        response = requests.post(
            f"{OLLAMA_HOST}/api/chat",
            json={"model": ollama_model, "messages": msgs},
            stream=True # Enable streaming to read line by line
        )
        response.raise_for_status()  # Raise an exception for bad status codes

        full_response_content = ""
        for chunk in response.iter_content(chunk_size=None):
            if chunk:
                try:
                    json_chunk = json.loads(chunk.decode('utf-8'))
                    if "message" in json_chunk and "content" in json_chunk["message"]:
                        full_response_content += json_chunk["message"]["content"]
                except json.JSONDecodeError:
                    print(f"DEBUG: Could not decode JSON chunk: {chunk.decode('utf-8')}")
                    # If it's not a valid JSON, it might be the extra data
                    # For now, we'll just print it and continue, but this is where the problem lies.
                    pass
        return full_response_content.strip()
    except requests.exceptions.RequestException as e:
        # Handle connection errors and other request issues
        return f"Error communicating with Ollama: {e}"

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
