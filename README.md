# Customer Support Ticket Auto-Classifier and Responder

End-to-end AI system for intelligent ticket routing and response generation using deep learning and NLP.

## Project Overview

This project predicts a support ticket category, estimates urgency, extracts intent, and generates a suggested response. It combines a deep learning model with an NLP pipeline and presents the result in a Flask web application.

## What I Built

- A PyTorch BiLSTM model for ticket classification and urgency scoring.
- A data preparation pipeline for cleaning text, building vocabulary, and creating train/validation/test splits.
- An NLP module for intent extraction and response generation.
- A Flask web app that displays the deep learning output and NLP output together.
- A consolidated repository structure with organized source code, data, and model files.

## Architecture

The application uses two cooperating pipelines:

1. Deep Learning pipeline
   - classifies the support ticket
   - predicts urgency as a score between 0 and 1

2. NLP pipeline
   - maps the ticket to a simpler support group
   - extracts the subject and action
   - generates a response template

The web layer combines both results and displays them side by side.

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
├── LICENSE
├── CONTRIBUTING.md
└── .github/
    └── workflows/
        └── ci.yml
```

## Deep Learning Details

The deep learning module uses a BiLSTM architecture with:

- an embedding layer,
- a 2-layer bidirectional LSTM,
- a classification head,
- a regression head.

Training uses:

- class-weighted cross-entropy for category prediction,
- mean squared error for urgency,
- Adam optimizer,
- gradient clipping.

The best checkpoint is stored in `models/best_model.pth`.

## NLP Details

The NLP module includes:

- zero-shot classification with `facebook/bart-large-mnli`,
- spaCy-based intent extraction,
- category-based response templates.

It produces:

- mapped support category,
- subject,
- action,
- suggested reply.

## Web Application

The Flask app lets a user paste a ticket and see:

- the deep learning category prediction,
- the urgency score,
- the NLP subject and action,
- the generated response.

Run the app with:

```bash
python app.py
```

Then open `http://127.0.0.1:5000`.

## Setup

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

### 3. Run the application

```bash
python app.py
```

## Training and Data Preparation

To rebuild the splits and vocabulary:

```bash
python -m src.dl.prepare_data
```

To retrain the model:

```bash
python -m src.dl.train
```

## Example Usage

### Python API

```python
from src.dl.predict import integrated_predict

result = integrated_predict("I was charged twice for my order!")
print(result)
```

### Web UI

1. Start the server with `python app.py`.
2. Open the local address in your browser.
3. Paste a customer support ticket.
4. View the DL prediction, urgency, intent, and suggested response.


## Technologies Used

- Python
- PyTorch
- Flask
- Transformers
- spaCy
- pandas
- scikit-learn

## License

MIT License. See [LICENSE](LICENSE) for details.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.
