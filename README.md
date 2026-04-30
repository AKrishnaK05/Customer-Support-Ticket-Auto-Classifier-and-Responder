# Customer Support Ticket Auto-Classifier and Responder

End-to-end AI system for intelligent ticket routing and response generation using deep learning and NLP.

## Architecture

**Dual-Model Pipeline**:
1. **Deep Learning (BiLSTM)**: Classification + urgency scoring with class-weighted loss
2. **NLP (BART + spaCy)**: Intent extraction + empathetic response generation

Both models run together to produce comprehensive ticket analysis.

## Project Structure

```text
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
.gitignore           # Standard Python ignores + tclass, data/*, models/*
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

```

## CI/CD

GitHub Actions workflow (`.github/workflows/ci.yml`) runs on every push/PR to verify code compiles.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License — see [LICENSE](LICENSE) for details.

---

**Built with**: PyTorch, Flask, Transformers, spaCy, pandas, scikit-learn
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

# Customer Support Ticket Auto-Classifier and Responder

This project is an end-to-end customer support assistant that analyzes a ticket, predicts its support category, estimates urgency, extracts intent, and generates a suggested response. The final repository is consolidated into a single clean structure with deep learning, NLP, and web UI components working together.

## Project Summary

The system combines two approaches:

1. Deep learning for ticket classification and urgency prediction.
2. NLP for intent extraction, category mapping, and response generation.

These are exposed through a Flask web app that shows both outputs side by side.

## What I Built

I developed and integrated the following parts of the system:

- A PyTorch BiLSTM model for ticket classification and urgency scoring.
- A preprocessing and data preparation pipeline for cleaning text, building vocabulary, and splitting data.
- An NLP pipeline using spaCy and a transformer-based zero-shot classifier.
- A response generator that creates a short support reply based on the predicted category and extracted intent.
- A modern Flask web interface to present the deep learning and NLP results together.
- A complete repository cleanup and consolidation into one organized codebase.

## Final Repository Changes

The repository was reorganized so the code is easier to maintain and ready for GitHub submission:

- Old root wrapper files were removed.
- The partner NLP code was merged into `src/nlp/`.
- Data files were moved into `data/`.
- The trained checkpoint was moved into `models/`.
- The root `app.py` was kept as the single entry point.
- Imports were updated to use the new package layout.

## Architecture

The application follows a dual-model design:

### Deep Learning Pipeline

- Input ticket text is cleaned and tokenized.
- The token sequence is passed through an embedding layer.
- A 2-layer bidirectional LSTM learns the sequence representation.
- The model produces two outputs:
   - a support category prediction
   - an urgency score in the range $[0, 1]$

### NLP Pipeline

- A zero-shot classifier maps the ticket to a simpler support group.
- spaCy is used to extract a subject and action from the text.
- A template-based responder creates a suggested reply.

### Web Layer

- Flask receives user input from the browser.
- The app calls the integrated prediction pipeline.
- Results are shown in a polished two-panel layout.

## Repository Structure

```text
.
├── app.py
├── data/
│   ├── cleaned_customer_support_tickets.csv
│   ├── customer_support_tickets.csv
│   ├── label_mapping.csv
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
│   └── vocab.json
├── models/
│   └── best_model.pth
├── src/
│   ├── dl/
│   │   ├── dataset.py
│   │   ├── model.py
│   │   ├── predict.py
│   │   ├── prepare_data.py
│   │   ├── train.py
│   │   └── utils.py
│   ├── nlp/
│   │   ├── classifier.py
│   │   ├── extractor.py
│   │   └── responder.py
│   └── web/
│       └── app.py
├── static/
├── templates/
├── tests/
├── requirements.txt
├── README.md
├── LICENSE
├── CONTRIBUTING.md
└── .github/
      └── workflows/
            └── ci.yml
```

## Deep Learning Details

The ticket classifier is a PyTorch-based BiLSTM model.

### Main components

- Embedding layer with token vectors.
- Bidirectional LSTM with two layers.
- Classification head for issue category.
- Regression head for urgency estimation.

### Training strategy

- Class-balanced loss to handle label imbalance.
- Mean squared error for urgency prediction.
- Adam optimizer.
- Gradient clipping for training stability.
- Best checkpoint saved as `models/best_model.pth`.

## NLP Details

The NLP system is used to make the output more human-readable and useful for support workflows.

### Components

- `classifier.py`: zero-shot classification with `facebook/bart-large-mnli`.
- `extractor.py`: subject and action extraction using spaCy.
- `responder.py`: template-based response creation.

### Output

The NLP module produces:

- mapped support category
- subject phrase
- action phrase
- suggested reply text

## Web Application

The Flask application provides a simple interactive interface.

### UI behavior

- The user enters a customer support ticket.
- The backend runs the combined prediction pipeline.
- The page displays:
   - the deep learning category prediction
   - the urgency estimate
   - the NLP-derived subject and action
   - the generated response

### Entry point

- Run `python app.py` to launch the application.

## How the System Works

1. The user submits a ticket.
2. The text is cleaned and normalized.
3. The deep learning model predicts the support category and urgency.
4. The NLP module extracts intent and creates a support response.
5. The Flask app renders both results in the browser.

## Setup Instructions

### 1. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Run the app

```bash
python app.py
```

Then open `http://127.0.0.1:5000` in your browser.

## Training and Rebuilding Data

If you want to rebuild the vocabulary and the train/validation/test splits, run:

```bash
python -m src.dl.prepare_data
```

If you want to retrain the deep learning model, run:

```bash
python -m src.dl.train
```

## Example Usage

### Python API

```python
from src.dl.predict import integrated_predict

result = integrated_predict("I was charged twice for my order")
print(result)
```

The output includes:

- `dl_category`
- `category`
- `urgency`
- `subject`
- `action`
- `response`

### Web UI

1. Start the server with `python app.py`.
2. Open the local URL in a browser.
3. Paste a support ticket.
4. View the classification, urgency, intent, and generated response.

## Testing

Run the smoke test:

```bash
python tests/smoke_test.py
```

You can also run a compile check:

```bash
python -m py_compile src/dl/*.py src/nlp/*.py src/web/*.py
```

## What Was Cleaned Up

The repository was normalized into one consolidated project by:

- removing duplicate root-level wrapper scripts
- removing the separate `customer_support_nlp` folder
- placing NLP code inside `src/nlp/`
- storing datasets in `data/`
- storing the trained checkpoint in `models/`
- keeping only one root application entry point

## Tools and Libraries Used

- PyTorch
- Flask
- Transformers
- spaCy
- pandas
- scikit-learn

## Future Improvements

- Add model evaluation metrics to the UI.
- Save prediction logs for auditing.
- Add API endpoints for external integration.
- Expand intent templates for more ticket categories.
- Add Docker support for easy deployment.

## License

MIT License. See [LICENSE](LICENSE) for details.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.
