"""Simple local smoke test for the integrated pipeline.
This script is intended to be run locally (it may require heavy deps).
"""
from predict import integrated_predict
import json

SAMPLES = [
    "I was charged twice for my subscription and need a refund",
    "My app crashes when I upload a file",
    "Someone accessed my account without permission"
]

results = []
for s in SAMPLES:
    try:
        out = integrated_predict(s)
    except Exception as e:
        out = {"error": str(e)}
    out["text"] = s
    results.append(out)

print(json.dumps(results, indent=2))
