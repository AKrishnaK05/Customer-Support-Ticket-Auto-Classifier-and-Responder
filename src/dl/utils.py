import re

MAX_LEN = 160
VOCAB_SIZE = 10000
EMBED_DIM = 100
HIDDEN_DIM = 320


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    text = re.sub(r"\bdear customer support team\b", "", text)
    text = re.sub(r"\bdear customer service team\b", "", text)
    text = re.sub(r"\bi hope this message finds you well\b", "", text)
    text = re.sub(r"\bi hope this message reaches you well\b", "", text)
    text = re.sub(r"\bi am writing to report\b", "", text)
    text = re.sub(r"\bi am reaching out to report\b", "", text)
    text = re.sub(r"\bi am writing to request\b", "", text)
    text = re.sub(r"\bi am reaching out to request\b", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def text_to_sequence(text, vocab, max_len=MAX_LEN):
    tokens = text.split()
    seq = [vocab.get(word, 1) for word in tokens]
    if len(seq) < max_len:
        seq += [0] * (max_len - len(seq))
    else:
        seq = seq[:max_len]
    return seq
