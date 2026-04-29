from transformers import pipeline

classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

def classify(text, labels):
    result = classifier(text, labels)
    return result["labels"][0], result["scores"][0]
