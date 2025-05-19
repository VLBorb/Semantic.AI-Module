# Semantic.AI-Module

This project provides a semantic and grammatical layer around GPT-2. It supports two execution modes:

- **TRAIN** – prepare vocabulary files from text corpora and fine‑tune GPT‑2
- **INFERENCE** – load the trained model and interact with it through a conversation loop

## Setup

Install dependencies and spaCy language model:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Usage

Run in inference mode (default):

```bash
python semantic_ai.py --mode INFERENCE
```

To prepare data and fine‑tune the model:

```bash
python semantic_ai.py --mode TRAIN
```

The base directory for data and models can be changed with the `SEMANTIC_AI_BASE_DIR` environment variable.
