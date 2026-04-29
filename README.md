# Customer Support Ticket Auto-Classifier and Responder

End-to-end AI system for intelligent ticket routing and response generation using deep learning and NLP.

## Architecture

**Dual-Model Pipeline**:
1. **Deep Learning (BiLSTM)**: Classification + urgency scoring with class-weighted loss
2. **NLP (BART + spaCy)**: Intent extraction + empathetic response generation

Both models run together to produce comprehensive ticket analysis.

## Project Structure

```
src/
├── dl/              # Deep Learning module (PyTorch BiLSTM)
│   ├── model.py     # TicketLSTM: 2-layer BiLSTM classifier + regressor
│   ├── dataset.py   # TicketDataset: PyTorch data loader
│   ├── train.py     # Training loop with class weighting & gradient clipping
│   ├── predict.py   # Inference + integrated pipeline (DL + NLP)
│   ├── utils.py     # Text cleaning, tokenization, constants
│   ├── prepare_data.py  # Vocabulary building, train/val/test splits
│   └── __init__.py
├── web/             # Flask web application
│   ├── app.py       # Flask routes, dual-panel UI
│   └── __init__.py
├── nlp/             # NLP module (Transformers + spaCy)
│   ├── classifier.py    # Zero-shot classification (facebook/bart-large-mnli)
│   ├── extractor.py # Intent extraction (spaCy NER + dep parsing)
│   ├── responder.py # Template-based response generation
│   └── __init__.py
└── __init__.py

data/                # Datasets & vocabulary (Git-tracked via .gitkeep)
├── cleaned_customer_support_tickets.csv
├── label_mapping.csv
├── vocab.json
├── train.csv, val.csv, test.csv
└── .gitkeep

models/              # Model checkpoints (Git-tracked via .gitkeep)
├── best_model.pth
└── .gitkeep

templates/           # Jinja2 HTML (modern SaaS UI)
├── index.html       # Dual-panel layout (DL left, NLP right)
└── ...

static/              # CSS & assets
├── style.css        # Inter font, gradients, animations
└── ...

tests/               # Integration tests
├── smoke_test.py
└── ...

app.py               # Root entry point (imports from src.web.app)
requirements.txt     # All dependencies (PyTorch, Flask, Transformers, spaCy, etc.)
.gitignore           # Standard Python ignores + tclass/, data/*, models/*
.github/workflows/ci.yml  # GitHub Actions CI
LICENSE              # MIT
CONTRIBUTING.md      # Contribution guidelines
README.md            # This file
```

## Quickstart

### 1. Setup

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Run Flask Demo

```bash
python app.py
# Visit http://127.0.0.1:5000
```

Paste a customer support ticket and see both DL and NLP predictions side-by-side.

### 4. Training (Optional)

To retrain the deep learning model:

```bash
python -m src.dl.train
```

Preprocessed data (train/val/test splits) should already exist in `data/`.

## Model Details

**Deep Learning (BiLSTM)**:
- Embedding layer (GloVe-style, EMBED_DIM=100)
- 2-layer Bidirectional LSTM (HIDDEN_DIM=320)
- Dual heads: classification (7 categories) + urgency regression [0–1]
- Class-weighted cross-entropy loss + MSE loss (0.01 weight)
- Batch size 32, Adam optimizer, gradient clipping

**NLP**:
- Zero-shot classifier: facebook/bart-large-mnli (14 labels)
- Intent extraction: spaCy NER + dependency parsing + action/subject mapping
- Response generation: Category-specific templates with subject/action substitution

## Usage Examples

### Python API

```python
from src.dl.predict import integrated_predict

result = integrated_predict("I was charged twice for my order!")

print(result)
# {
#   "dl_category": "Billing & Returns",
#   "category": "billing",
#   "urgency": 0.78,
#   "subject": "payment",
#   "action": "refund",
#   "response": "We're sorry for the issue with payment. We will process your refund shortly."
# }
```

### Web UI

1. Run `python app.py`
2. Open `http://127.0.0.1:5000`
3. Paste a ticket in the textarea
4. View dual-model predictions:
   - **Left panel (DL)**: Category, urgency bar, priority badge
   - **Right panel (NLP)**: Mapped category, subject, action, suggested response

## Testing

Run smoke tests:

```bash
python tests/smoke_test.py
```

Compile check (all Python files):

```bash
python -m py_compile src/dl/*.py src/web/*.py src/nlp/*.py
```

## CI/CD

GitHub Actions workflow (`.github/workflows/ci.yml`) runs on every push/PR to verify code compiles.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License — see [LICENSE](LICENSE) for details.

---

**Built with**: PyTorch, Flask, Transformers, spaCy, pandas, scikit-learn
