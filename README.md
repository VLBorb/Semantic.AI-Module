#codex/add-requirements-txt-with-dependencies
# Semantic.AI-Module
# By. V. Lucian Borbeleac
codex/set-up-semantic-ai-system-configuration

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
=======
#codex/set-up-pytest-and-add-unit-tests
Semantic-AI: Grammatical Semantics Extension Module This module extends the core Semantic-AI system with a Grammatical Semantics Layer (GSL) that enables syntactic coherence across generated sequences.

## Running Tests

The project uses `pytest` for unit tests. To run the test suite locally:

```bash
pip install pytest
pytest
```

All tests live in the `tests/` directory.
=======

Semantic-AI: Grammatical Semantics Extension Module. This module extends the core Semantic-AI system with a Grammatical Semantics Layer (GSL) that enables syntactic coherence across generated sequences.

## Installation

Install the required Python packages before using the module:

```bash
pip install -r requirements.txt
```

# Semantic AI with Llama 2: A Cognitive Architecture

This project implements a novel semantic artificial intelligence (AI) system that combines the power of large language models (LLMs), specifically Llama 2, with a structured semantic representation. The goal is to create an AI capable of more controlled, coherent, and explainable text processing and generation.

## Core Features

* **Semantic Encoding:**
    * Transforms natural language words into 12-digit numeric codes.
    * These codes capture linguistic features: Part of Speech (POS), subcategory, polarity, abstractness, semantic domain, register, rhyme, and a local identifier.

* **Semantic Validation:**
    * Analyzes sequences of semantic codes using predefined grammatical and semantic rules.
    * Ensures the validity of word sequences based on POS succession, polarity consistency, register consistency, etc.

* **Controlled Text Generation:**
    * Fine-tunes the Llama 2 model to generate text that adheres to the semantic codes and validation rules.
    * Enables the generation of text with specific stylistic and semantic properties.

* **Epistemic Reasoning:**
    * Includes a basic deductive cognition engine to perform simple logical inferences.
    * Incorporates a belief module to simulate the formation and updating of beliefs based on context.

* **Ethical Constraints:**
    * Implements an ethical constitution that validates generated text against predefined rules to prevent harmful or inappropriate content.

* **Contextual Memory:**
    * Utilizes a SQLite database to store and retrieve contextual information, enabling the system to maintain a history of interactions.

* **Traceability and Logging:**
    * Provides comprehensive logging of system operations, including input processing, code generation, validation results, and ethical evaluations.

## Architecture

The system is designed with a modular architecture:

1. **Vocabulary Management:**
    * Loads and manages the mapping between words and semantic codes.
    * Includes functionality to generate and save the semantic vocabulary.

2. **Semantic Processing:**
    * Encodes text into semantic code sequences.
    * Validates code sequences based on grammatical and semantic rules.
    * Calculates a semantic score to evaluate the coherence of code sequences.

3. **Language Generation:**
    * Fine-tunes a Llama 2 model on text data augmented with semantic codes.
    * Generates text from sequences of semantic codes.

4. **Cognitive Modules:**
    * Deductive Cognition Engine: Performs basic logical inferences.
    * Belief Module: Simulates belief formation and updating.

5. **Ethical Governance:**
    * Ethical Constitution: Enforces ethical guidelines on generated text.

6. **Contextual Awareness:**
    * Contextual Memory: Stores and retrieves past interactions.

7. **Evaluation:**
    * Evaluation Metrics: Calculates various metrics to assess the quality of generated text (BLEU, ROUGE, lexical diversity, logical consistency, semantic score).

## Technology Stack

* **Programming Language:** Python
* **Large Language Model:** Llama 2
* **NLP Libraries:** spaCy, NLTK, phonetics
* **Deep Learning Framework:** PyTorch
* **Transformers Library:** Hugging Face Transformers
* **Database:** SQLite

## Execution Modes

The system supports two execution modes:

* **TRAIN:**
    * Used for initial data preparation and model fine-tuning.
    * Downloads and processes the corpus, generates the semantic vocabulary, and fine-tunes the Llama 2 model.

* **INFERENCE:**
    * Used for running the semantic conversation system.
    * Loads the pre-trained model, vocabularies, and matrices, and enables interactive conversation.

## Setup and Usage

(These instructions can be expanded to cover dependency installation, configuration, and how to run the system. Adjust to match your environment.)

## Contributions

(Provide guidelines for contributing to the project and acknowledge contributors.)

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
main
