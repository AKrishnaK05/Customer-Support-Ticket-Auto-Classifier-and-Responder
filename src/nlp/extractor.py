import spacy

nlp = spacy.load("en_core_web_sm")

KEY_ACTIONS = ["refund", "cancel", "fix", "update", "replace", "block"]

# 🔥 Map bad verbs → meaningful actions
ACTION_MAP = {
    "crash": "fix",
    "fail": "fix",
    "error": "fix",
    "not work": "fix",
    "break": "fix",
    "issue": "fix",
    "bug": "fix",
    "charge": "refund"
}

KEY_SUBJECTS = ["charge", "payment", "transaction", "account", "subscription", "app"]


def extract_intent(text):
    doc = nlp(text)

    subject = ""
    action = ""

    # 🔹 Step 1: keyword-based action
    for token in doc:
        if token.text.lower() in KEY_ACTIONS:
            action = token.text

    # 🔹 Step 2: smarter verb mapping
    if action == "":
        for token in doc:
            lemma = token.lemma_.lower()

            if lemma in ACTION_MAP:
                action = ACTION_MAP[lemma]
                break

            if token.pos_ == "VERB":
                action = lemma
                break

    # 🔹 Step 3: subject extraction
    STOP_SUBJECTS = ["it", "this", "that"]

    # 🔹 Step 3A: keyword-based subject
    for token in doc:
        if token.lemma_.lower() in KEY_SUBJECTS:
            subject = token.lemma_
            break

    # 🔹 Step 3B: noun fallback
    if subject == "":
        for token in doc:
            if (
                token.pos_ == "NOUN"
                and token.text.lower() not in STOP_SUBJECTS
                and token.text.lower() not in KEY_ACTIONS
            ):
                subject = token.text
                break

    # 🔹 Step 3C: dependency fallback
    if subject == "":
        for token in doc:
            if (
                token.dep_ in ["dobj", "pobj"]
                and token.text.lower() not in STOP_SUBJECTS
                and token.text.lower() not in KEY_ACTIONS
            ):
                subject = token.text
                break

    # 🔹 Final fallback
    if subject == "":
        subject = "your request"

    return subject, action
