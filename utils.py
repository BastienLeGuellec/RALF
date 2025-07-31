import tiktoken

from config import OPENAI_MODEL

try:
    tok_enc = tiktoken.encoding_for_model(OPENAI_MODEL)
    n_tokens = lambda text: len(tok_enc.encode(text))
except ImportError:
    n_tokens = lambda text: max(1, len(text.split()) // 0.75)
