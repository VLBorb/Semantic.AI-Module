# ================================================================
# Unified Semantic-AI-Epistemic with GSL Integration (Refined)
# Optimized for Server Deployment (Run in INFERENCE mode)
# Separate TRAIN mode for data prep and model training.
# ================================================================

# --- 0. Configuration & Mode Selection ---
import os
import sys
import json
import time
import argparse # For command-line arguments
import logging
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define execution modes
MODES = ["INFERENCE", "TRAIN"]

# --- Configuration (Modify as needed or use environment variables) ---
# General Paths
BASE_DIR = os.getenv("SEMANTIC_AI_BASE_DIR", "/app/semantic_ai") # Base directory on server
MODEL_DIR = os.path.join(BASE_DIR, "model")
VOCAB_DIR = os.path.join(BASE_DIR, "vocab")
DATA_DIR = os.path.join(BASE_DIR, "data")
LOG_DIR = os.path.join(BASE_DIR, "logs")
CORPUS_DIR = os.path.join(DATA_DIR, "corpus") # For downloaded corpus in TRAIN mode

# File Paths (Ensure these files exist in INFERENCE mode)
SEMANTIC_VOCAB_FILE = os.path.join(VOCAB_DIR, "semantic_vocab_12digits_new.json")
GRAMMAR_VOCAB_FILE = os.path.join(VOCAB_DIR, "semantic_grammar_codes.json")
PROB_MATRIX_FILE = os.path.join(VOCAB_DIR, "prob_matrix.json")
DB_PATH = os.path.join(DATA_DIR, "context_memory.db")
TRACEABILITY_LOG_PATH = os.path.join(LOG_DIR, "traceability")
TRAINING_CORPUS_FILE = os.path.join(DATA_DIR, "train.txt") # Generated in TRAIN mode

# Model & Training Config
MODEL_NAME = "gpt2-medium" # Base model for training
FINETUNED_MODEL_PATH = os.path.join(MODEL_DIR, "gpt2-semantic-finetuned") # Where fine-tuned model is saved/loaded
TRAINING_EPOCHS = 3
TRAINING_BATCH_SIZE = 2
FP16_TRAINING = True # Use mixed precision if GPU supports it

# Epistemic Module Config
BELIEF_DECAY_RATE = 0.05
MAX_BELIEFS = 1000
DEDUCTION_MAX_HOPS = 3

# Other Config
CORPUS_CHUNK_SIZE = 700000
MAX_VOCAB_WORDS = 50000
SEMANTIC_CODE_LENGTH = 12
GPU_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Create directories if they don't exist (important for first run/server setup)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(VOCAB_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(TRACEABILITY_LOG_PATH, exist_ok=True)
os.makedirs(CORPUS_DIR, exist_ok=True)
os.makedirs(FINETUNED_MODEL_PATH, exist_ok=True) # Ensure target training dir exists

# --- Argument Parser for Mode ---
parser = argparse.ArgumentParser(description="Run Semantic AI System")
parser.add_argument(
    "--mode",
    type=str,
    choices=MODES,
    default="INFERENCE",
    help=f"Execution mode: {', '.join(MODES)} (default: INFERENCE)"
)
args = parser.parse_args()
EXECUTION_MODE = args.mode

logger.info(f"[INFO] Running in {EXECUTION_MODE} mode.")
logger.info(f"[INFO] Using device: {GPU_DEVICE}")
logger.info(f"[INFO] Base directory: {BASE_DIR}")

# --- 1. Install Libraries (Commented out - manage via requirements.txt) ---
# Ensure these are installed on the server:
# pip install -U spacy nltk phonetics transformers==4.28.0 torch datasets wikiextractor rouge-score
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 # Adjust cuXXX for your server CUDA
# python -m spacy download en_core_web_sm
# Make sure NLTK data is downloaded: python -m nltk.downloader wordnet omw-1.4

# --- Essential Imports ---
import spacy
import nltk
import phonetics # Note: phonetics library might not be directly used later, review if needed
import requests
import re
import hashlib
import sqlite3
from collections import Counter, deque
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset, load_dataset # Modified for handling text file directly
from nltk.corpus import wordnet as wn
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import random
import numpy as np

from typing import Dict, List, Optional, Callable, Tuple, Any

from semantic_models import (
    DeductiveCognitionEngine,
    BeliefModule,
    ContextualMemory,
    EthicalConstitution,
    TraceabilityLogger,
    EvaluationMetrics,
)

# --- Download NLTK data if not present ---
try:
    nltk.data.find('corpora/wordnet.zip')
except nltk.downloader.DownloadError:
    logger.info("[INFO] Downloading NLTK wordnet data...")
    nltk.download('wordnet', quiet=True)
try:
    nltk.data.find('corpora/omw-1.4.zip')
except nltk.downloader.DownloadError:
    logger.info("[INFO] Downloading NLTK omw-1.4 data...")
    nltk.download('omw-1.4', quiet=True)

# --- 2. Utility Functions (Consolidated & Refined) ---

def is_valid_semantic_code(code: Any) -> bool:
    """Checks if a code is a valid semantic code string."""
    return isinstance(code, str) and len(code) == SEMANTIC_CODE_LENGTH and code.isdigit()

def is_valid_pos_succession(pos1: str, pos2: str, succession_matrix: Dict[str, List[str]]) -> bool:
    """Checks if POS tag pos2 can follow pos1 based on the succession matrix."""
    return pos2 in succession_matrix.get(pos1, [])

def check_full_pos_sequence(codes: List[str], succession_matrix: Dict[str, List[str]]) -> bool:
    """Checks if the entire sequence of POS tags is valid."""
    if not codes:
        return True # Empty sequence is considered valid

    valid_format_codes = [c for c in codes if is_valid_semantic_code(c)]
    if len(valid_format_codes) != len(codes):
         logger.info("[POS Check] Warning: Sequence contains invalid or non-semantic codes.")
         return False # Sequence with invalid codes is not valid

    for i in range(len(valid_format_codes) - 1):
        pos1 = valid_format_codes[i][:2]
        pos2 = valid_format_codes[i+1][:2]
        if not is_valid_pos_succession(pos1, pos2, succession_matrix):
            # print(f"[POS Check] Invalid succession: {pos1} -> {pos2}") # Optional debug
            return False
    return True

def load_json_file(file_path: str, description: str) -> Optional[Any]:
    """Loads data from a JSON file with error handling."""
    if not os.path.exists(file_path):
        logger.error(f"[ERROR] {description} file not found at: {file_path}")
        return None
    try:
        with open(file_path, "r", encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"[INFO] Successfully loaded {description} from: {file_path}")
        return data
    except json.JSONDecodeError as e:
        logger.error(f"[ERROR] Failed to decode JSON from {description} file: {file_path}. Error: {e}")
        return None
    except Exception as e:
        logger.error(f"[ERROR] Unexpected error loading {description} file: {file_path}. Error: {e}")
        return None

def load_semantic_vocab(vocab_file: str) -> Tuple[Optional[Dict[str, str]], Optional[Dict[str, str]], Optional[List[Dict]]]:
    """Loads the semantic vocabulary file."""
    vocab_data = load_json_file(vocab_file, "Semantic Vocabulary")
    if vocab_data is None:
        return None, None, None

    word_to_code: Dict[str, str] = {}
    code_to_word: Dict[str, str] = {}
    malformed_entries = 0
    for entry in vocab_data:
        word = entry.get("word")
        code = entry.get("semantic_code")
        if isinstance(word, str) and is_valid_semantic_code(code):
            word_to_code[word] = code
            # Handle potential duplicate codes (last one wins, or choose based on frequency if available)
            if code in code_to_word:
                 # print(f"[Vocab Load] Warning: Duplicate semantic code '{code}' for words '{code_to_word[code]}' and '{word}'. Using '{word}'.")
                 pass
            code_to_word[code] = word
        else:
            malformed_entries += 1

    if malformed_entries > 0:
         logger.warning(f"[WARN] Found {malformed_entries} malformed entries in semantic vocabulary file.")

    if not word_to_code:
         logger.error("[ERROR] No valid entries found in the semantic vocabulary.")
         return None, None, None

    return word_to_code, code_to_word, vocab_data

def load_grammar_vocab(vocab_file: str) -> Optional[Dict[str, str]]:
    """Loads grammar codes (GSL) from semantic_grammar_codes.json."""
    grammar_data = load_json_file(vocab_file, "Grammar (GSL) Vocabulary")
    if grammar_data is None:
        return None

    grammar_codes: Dict[str, str] = {}
    malformed_entries = 0
    for entry in grammar_data:
        word = entry.get("word")
        code = entry.get("grammar_code")
        # Add more validation for grammar code format if known
        if isinstance(word, str) and isinstance(code, str):
            grammar_codes[word] = code
        else:
             malformed_entries +=1

    if malformed_entries > 0:
         logger.warning(f"[WARN] Found {malformed_entries} malformed entries in grammar vocabulary file.")

    if not grammar_codes:
        logger.error("[ERROR] No valid entries found in the grammar vocabulary.")
        return None

    return grammar_codes

def text_to_codes(text: str, word_to_code: Dict[str, str]) -> Tuple[List[str], List[str]]:
    """Converts text to a list of semantic codes, returning codes and unknown words."""
    if not isinstance(text, str):
        logger.error("[ERROR] Input text is not a string.")
        return [], []
    words = text.lower().split()
    codes = [word_to_code.get(w, "N/A") for w in words]
    unknown_words = [words[i] for i, code in enumerate(codes) if code == "N/A"]
    return codes, unknown_words

def codes_to_text(codes: List[str], code_to_word: Dict[str, str]) -> str:
    """Converts a list of semantic codes back to text."""
    return " ".join([code_to_word.get(c, "[?]") for c in codes if c != "N/A"])

# Global cache for semantic_score
semantic_score_cache: Dict[Tuple[str, ...], float] = {}

def semantic_score(codes: List[str], succession_matrix: Dict[str, List[str]]) -> float:
    """Calculates a score based on internal consistency of semantic codes. Uses caching."""
    valid_codes = [c for c in codes if is_valid_semantic_code(c)]

    cache_key = tuple(valid_codes) # Use only valid codes for cache key
    if cache_key in semantic_score_cache:
        return semantic_score_cache[cache_key]

    score = 0
    max_score_per_pair = 5 # Max points per adjacent pair (POS, Polarity, Register, Domain, Abstract?)

    if len(valid_codes) < 2:
        return 0.0

    num_pairs = len(valid_codes) - 1
    total_max_score = num_pairs * max_score_per_pair

    if total_max_score == 0:
        return 0.0 # Avoid division by zero for single valid code sequences

    penalty = (len(codes) - len(valid_codes)) * max_score_per_pair # Penalize for N/A codes

    for i in range(num_pairs):
        c1 = valid_codes[i]
        c2 = valid_codes[i+1]

        # POS Succession Check (1 point)
        pos1, pos2 = c1[:2], c2[:2]
        if is_valid_pos_succession(pos1, pos2, succession_matrix):
            score += 1

        # Polarity Consistency (c[4]) (1 point if same or one is neutral '0')
        pol1, pol2 = c1[4], c2[4]
        if pol1 == pol2 or pol1 == '0' or pol2 == '0':
            score += 1

        # Abstract/Concrete Consistency (c[5]) (1 point if same)
        if c1[5] == c2[5]:
             score += 1

        # Domain Consistency (c[6:8]) (1 point if same or one is '00')
        dom1, dom2 = c1[6:8], c2[6:8]
        if dom1 == dom2 or dom1 == '00' or dom2 == '00':
            score += 1

        # Register Consistency (c[8]) (1 point if same or one is neutral '0')
        reg1, reg2 = c1[8], c2[8]
        if reg1 == reg2 or reg1 == '0' or reg2 == '0':
            score += 1

    # Normalize the score
    # Subtract penalty from achieved score, ensure non-negative, then normalize by max possible score
    normalized_score = max(0.0, score - penalty) / total_max_score * 100

    semantic_score_cache[cache_key] = normalized_score
    # Optional: Limit cache size
    if len(semantic_score_cache) > 10000:
         try: # Avoid error if cache is empty
             semantic_score_cache.pop(next(iter(semantic_score_cache)))
         except StopIteration:
             pass

    return round(normalized_score, 1)

def validate_sequence(codes: List[str], succession_matrix: Dict[str, List[str]], prob_matrix: Dict[str, Dict[str, float]],
                      check_polarity: bool = True, check_register: bool = True) -> Tuple[bool, str]:
    """Validates a sequence of semantic codes using POS succession and optional checks."""
    errors = []
    valid_codes = []
    has_invalid_format = False

    for i, code in enumerate(codes):
        if is_valid_semantic_code(code):
            valid_codes.append(code)
        else:
            errors.append(f"Invalid code format or N/A at index {i}: '{code}'")
            has_invalid_format = True

    if not valid_codes or has_invalid_format:
         # Decide if sequence with N/A or bad format is immediately invalid
         return False, "; ".join(errors) if errors else "Sequence contains invalid format codes."

    if len(valid_codes) < 2:
        return True, "Sequence valid (too short for pairwise checks)." # Or False if min length required

    for i in range(len(valid_codes) - 1):
        c1 = valid_codes[i]
        c2 = valid_codes[i+1]
        pos1, pos2 = c1[:2], c2[:2]
        pol1, pol2 = c1[4], c2[4]
        reg1, reg2 = c1[8], c2[8]

        # 1. POS Succession Check (using deterministic first, then probabilistic)
        if not is_valid_pos_succession(pos1, pos2, succession_matrix):
            # Check probabilistic matrix as fallback ONLY IF deterministic fails
            prob = prob_matrix.get(pos1, {}).get(pos2, 0.0)
            # Set a threshold for improbability (e.g., 1%)
            if prob < 0.01:
                 errors.append(f"POS violation ({i+1}): {pos1} → {pos2} (Prob={prob:.3f})")

        # 2. Polarity Shift Check (abrupt change between non-neutral polarities)
        if check_polarity and pol1 != '0' and pol2 != '0' and pol1 != pol2:
            errors.append(f"Polarity violation ({i+1}): {pol1} → {pol2}")

        # 3. Register Shift Check (abrupt change between non-neutral registers)
        if check_register and reg1 != '0' and reg2 != '0' and reg1 != reg2:
            errors.append(f"Register violation ({i+1}): {reg1} → {reg2}")

    if errors:
        return False, "; ".join(errors)
    return True, "Sequence valid"

def validate_grammar_sequence(text: str, codes: List[str], grammar_codes: Dict[str, str]) -> Tuple[bool, str]:
    """
    Validates the grammatical sequence using GSL codes.
    Assumes `codes` here are the *semantic* codes corresponding to the `text`.
    """
    errors = []
    words = text.lower().split()

    # Ensure text and codes align (essential for mapping words to GSL)
    if len(words) != len(codes):
        return False, f"Grammar Check Error: Word count ({len(words)}) doesn't match code count ({len(codes)}) for text: '{text}'"

    # Check if all words have GSL codes
    gsl_sequence = []
    for i, word in enumerate(words):
        gcode = grammar_codes.get(word)
        if gcode is None:
            errors.append(f"Grammar Check: Word '{word}' (index {i}) not found in GSL vocabulary.")
            # Decide: stop validation or continue? Continuing might give partial info.
            # For robustness, let's allow continuing but report missing words.
            gsl_sequence.append(None) # Placeholder for missing code
        # Optional: Add check for expected GSL code format if known (e.g., length, digits)
        # elif len(gcode) != 10: # Example check
        #     errors.append(f"Invalid GSL code format for '{word}': '{gcode}'")
        #     gsl_sequence.append(None)
        else:
            gsl_sequence.append(gcode)

    # --- Apply GSL Rules (Based on original code's logic - Requires understanding GSL code meaning) ---
    # This part is highly dependent on the specific definition of the GSL codes.
    # The original code had examples; add comments explaining them.

    for i in range(len(words)):
         # Skip checks if GSL code is missing for current or relevant adjacent words
         current_gsl = gsl_sequence[i]
         if current_gsl is None:
             continue

         # Example: Check number agreement (Noun following Verb)
         # Needs semantic POS codes AND GSL codes
         # Assuming: Semantic POS '01'=NOUN, '02'=VERB
         # Assuming: GSL c[0:2] = Number/Type? ('01'=Singular Noun?)
         if i > 0 and is_valid_semantic_code(codes[i]) and codes[i][:2] == "01" and \
            is_valid_semantic_code(codes[i-1]) and codes[i-1][:2] == "02":
             # Let's assume GSL gcode[0:2] encodes number/type info
             # Example Rule: If semantic codes show VERB -> NOUN, and GSL code for Verb indicates singular,
             # then GSL code for Noun must also indicate singular. (This is interpretation)
             # This requires specific knowledge of GSL format, which isn't fully defined.
             # Original check was: if gcode[0:2] != "01" and codes[i-1][:2] == "02":
             # This seems contradictory or requires clarification. Let's skip for now without clear GSL definition.
             # errors.append(f"Potential number disagreement at '{words[i]}'")
             pass # Skipping unclear rule

         # Example: Check syntactic role (Subject-Verb-Object structure)
         # Assuming: GSL c[2:4] = Syntactic Role ('01'=Subject, '02'=Object, '06'=Root?)
         role = current_gsl[2:4] if len(current_gsl) >= 4 else None
         if role:
              if i == 0 and role not in ["01", "06"]: # Expect Subject or Root at start
                  errors.append(f"Grammar Check: Invalid start role '{role}' for word '{words[i]}'")
              # Example Rule: If previous semantic code was Verb ('02'), expect Object role ('02') next.
              if i > 0 and is_valid_semantic_code(codes[i-1]) and codes[i-1][:2] == "02" and role != "02":
                  errors.append(f"Grammar Check: Expected Object role after verb, got '{role}' for word '{words[i]}'")

         # Example: Check verb agreement (Simplified)
         # Assuming: Semantic POS '02'=VERB
         # Assuming: GSL c[4:6] = Verb Form/Agreement ('01'-'04' valid?)
         if is_valid_semantic_code(codes[i]) and codes[i][:2] == "02":
              verb_form = current_gsl[4:6] if len(current_gsl) >= 6 else None
              if verb_form not in ["01", "02", "03", "04"]: # Example valid forms
                  errors.append(f"Grammar Check: Invalid verb form '{verb_form}' for word '{words[i]}'")

         # Example: Check word order (Simplified SVO expectation)
         # Assuming: GSL c[9] = Expected next type? ('1'=Object, '2'=Noun?) - Needs clarification!
         # This rule seems highly specific and potentially brittle.
         # order_expectation = current_gsl[9] if len(current_gsl) >= 10 else None
         # if order_expectation:
         #     if i < len(codes) - 1 and is_valid_semantic_code(codes[i+1]):
         #          next_pos = codes[i+1][:2]
         #          if order_expectation == "1" and next_pos != "02": # Expected Object (Role '02')? Or POS Noun ('01')? Clarify GSL meaning.
         #              errors.append(f"Grammar Check: Invalid word order at '{words[i]}', expected Object-like next?")
         #          if order_expectation == "2" and next_pos != "01": # Expected Noun (POS '01')?
         #               errors.append(f"Grammar Check: Invalid word order at '{words[i]}', expected Noun-like next?")
         pass # Skipping word order check due to unclear GSL definition

    if errors:
        return False, "; ".join(errors)
    return True, "Grammar sequence appears valid (based on available GSL rules)"

# --- 3. Epistemic Modules (Refined with error handling and utility usage) ---

# --- 4. Core Generation and Conversation Logic (Refined) ---

def generate_codes(seed_codes: List[str], model: GPT2LMHeadModel, tokenizer: GPT2TokenizerFast,
                   code_to_word: Dict[str, str], grammar_codes: Dict[str, str],
                   deductive_engine: DeductiveCognitionEngine, belief_module: BeliefModule,
                   prob_matrix: Dict[str, Dict[str, float]], succession_matrix: Dict[str, List[str]],
                   max_new_codes: int = 15, num_candidates_gpt: int = 3, current_cycle: int = 0) -> List[Dict]:
    """Generates candidate code sequences using GPT-2, Deduction, Beliefs, and Fallbacks."""
    candidates: List[Dict] = [] # List of {"codes": List[str], "text": str, "source": str}

    # --- 1. GPT-2 Generation ---
    if model and tokenizer and seed_codes: # Only generate if model and seed available
        # Prepare input for GPT-2 (use only valid codes)
        valid_seed_codes = [c for c in seed_codes if is_valid_semantic_code(c)]
        if valid_seed_codes:
            input_text = " ".join(valid_seed_codes)
            try:
                input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)

                # Generate multiple sequences
                # Use slightly diverse generation parameters if needed
                for i in range(num_candidates_gpt):
                     output_ids = model.generate(
                         input_ids,
                         max_length=input_ids.shape[1] + max_new_codes,
                         do_sample=True,
                         top_k=30, # Slightly increased diversity
                         top_p=0.95,
                         temperature=0.9 + i*0.05, # Vary temperature slightly
                         num_return_sequences=1,
                         pad_token_id=tokenizer.eos_token_id # Use eos_token_id as pad_token_id
                     )

                     # Decode generated sequence
                     full_generated_code_str = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                     # Extract only the newly generated part
                     newly_generated_code_str = full_generated_code_str.replace(input_text, "").strip()
                     generated_codes = newly_generated_code_str.split()

                     # Validate generated codes
                     valid_new_codes = [code for code in generated_codes if is_valid_semantic_code(code) and code in code_to_word]

                     if valid_new_codes:
                         full_sequence = valid_seed_codes + valid_new_codes
                         text = codes_to_text(full_sequence, code_to_word)

                         # Validate sequence grammar and semantics
                         seq_valid, seq_msg = validate_sequence(full_sequence, succession_matrix, prob_matrix)
                         gram_valid, gram_msg = validate_grammar_sequence(text, full_sequence, grammar_codes) # Use full sequence for grammar check

                         if seq_valid and gram_valid:
                              candidates.append({
                                  "codes": valid_new_codes, # Return only the *new* codes generated
                                  "text": codes_to_text(valid_new_codes, code_to_word), # Text for new codes only
                                  "source": "gpt2"
                                  })
                         # else: # Debugging
                         #     print(f"[Gen DEBUG] GPT cand rejected. Seq: {seq_valid} ({seq_msg}). Gram: {gram_valid} ({gram_msg})")
                         #     print(f"[Gen DEBUG] Text: '{text}'")
                         #     print(f"[Gen DEBUG] Codes: {full_sequence}")


            except Exception as e:
                logger.error(f"[ERROR] GPT-2 generation failed: {e}")
                # Potentially log the error with logger.log_error(...)

    # --- 2. Deductive Engine ---
    if deductive_engine and seed_codes:
         premise_text = codes_to_text(seed_codes, code_to_word) # Use text from original seed codes
         if premise_text:
              deduction = deductive_engine.deduce(premise_text, seed_codes)
              if deduction and deduction.get("code") != "N/A":
                   deduced_code = deduction["code"]
                   deduced_text = deduction["text"] # Full consequence text
                   # Simple text: use the word associated with the code
                   simple_text = code_to_word.get(deduced_code, "[?]")

                   # Validate the single deduced code and its grammar in context
                   full_sequence = seed_codes + [deduced_code]
                   text_for_validation = codes_to_text(seed_codes, code_to_word) + " " + simple_text

                   seq_valid, _ = validate_sequence(full_sequence, succession_matrix, prob_matrix)
                   gram_valid, _ = validate_grammar_sequence(text_for_validation, full_sequence, grammar_codes)

                   if seq_valid and gram_valid:
                       candidates.append({
                           "codes": [deduced_code],
                           "text": simple_text, # Use the single word for the candidate text
                           "source": "deduction"
                       })
                   # else: # Debugging
                   #     print(f"[Gen DEBUG] Deduction cand rejected. Seq: {seq_valid}. Gram: {gram_valid}")

    # --- 3. Belief Module ---
    if belief_module and seed_codes:
         # Evaluate beliefs based on current context (seed_codes)
         # Note: evaluate_belief updates internal strengths
         for prop in list(belief_module.beliefs.keys()): # Iterate over keys copy
              belief_module.evaluate_belief(prop, seed_codes, current_cycle)

         best_belief = belief_module.get_best_belief()
         if best_belief and best_belief.get("code") != "N/A":
              belief_code = best_belief["code"]
              belief_text = best_belief["proposition"] # Usually a single word/concept

              # Validate the single belief code and its grammar in context
              full_sequence = seed_codes + [belief_code]
              text_for_validation = codes_to_text(seed_codes, code_to_word) + " " + belief_text

              seq_valid, _ = validate_sequence(full_sequence, succession_matrix, prob_matrix)
              gram_valid, _ = validate_grammar_sequence(text_for_validation, full_sequence, grammar_codes)

              if seq_valid and gram_valid:
                    candidates.append({
                        "codes": [belief_code],
                        "text": belief_text,
                        "source": "belief"
                    })
                    # Optional: Boost belief strength slightly for being chosen?
                    # belief_module.update_belief(belief_text, best_belief['B'] + 0.1, seed_codes, current_cycle)
              # else: # Debugging
              #      print(f"[Gen DEBUG] Belief cand rejected. Seq: {seq_valid}. Gram: {gram_valid}")


    # --- 4. Probabilistic Fallback ---
    if not candidates and seed_codes and prob_matrix: # Only if no candidates yet
        last_valid_code = next((c for c in reversed(seed_codes) if is_valid_semantic_code(c)), None)
        if last_valid_code:
            last_pos = last_valid_code[:2]
            next_pos_probs = prob_matrix.get(last_pos, {})

            if next_pos_probs:
                pos_choices = list(next_pos_probs.keys())
                probabilities = list(next_pos_probs.values())
                # Normalize probabilities
                total_prob = sum(probabilities)
                if total_prob > 0:
                    normalized_probs = [p / total_prob for p in probabilities]

                    # Try a few times to find a valid fallback
                    for _ in range(5): # Try up to 5 times
                        chosen_pos = np.random.choice(pos_choices, p=normalized_probs)
                        # Find words with this POS
                        possible_codes = [code for code, word in code_to_word.items()
                                          if is_valid_semantic_code(code) and code.startswith(chosen_pos)]
                        if possible_codes:
                             fallback_code = random.choice(possible_codes)
                             fallback_text = code_to_word.get(fallback_code, "[?]")

                             # Validate fallback code in context
                             full_sequence = seed_codes + [fallback_code]
                             text_for_validation = codes_to_text(seed_codes, code_to_word) + " " + fallback_text

                             seq_valid, _ = validate_sequence(full_sequence, succession_matrix, prob_matrix)
                             gram_valid, _ = validate_grammar_sequence(text_for_validation, full_sequence, grammar_codes)

                             if seq_valid and gram_valid:
                                  candidates.append({
                                      "codes": [fallback_code],
                                      "text": fallback_text,
                                      "source": "fallback"
                                  })
                                  # print(f"[Gen DEBUG] Fallback candidate accepted: {fallback_text}")
                                  break # Stop after finding one valid fallback
                        # else: # Debugging
                        #      print(f"[Gen DEBUG] Fallback cand rejected. Seq: {seq_valid}. Gram: {gram_valid}")

    # print(f"[DEBUG] Generated {len(candidates)} candidates.")
    # Remove duplicates (based on text AND codes) - simple approach
    unique_candidates = []
    seen = set()
    for cand in candidates:
         cand_key = (cand['text'], tuple(cand['codes']))
         if cand_key not in seen:
             unique_candidates.append(cand)
             seen.add(cand_key)

    return unique_candidates

def rank_candidates(candidates: List[Dict], evaluation_metrics: EvaluationMetrics,
                    reference_text: str, seed_codes: List[str],
                    succession_matrix: Dict[str, List[str]]) -> Optional[Dict]:
    """Ranks candidates based on aggregate score and semantic score."""
    if not candidates:
        return None

    scored_candidates = []
    full_seed_text = codes_to_text(seed_codes, evaluation_metrics.word_to_code) # Need code_to_word here

    for cand in candidates:
        # Aggregate score needs full response text and full codes
        full_response_text = (full_seed_text + " " + cand["text"]).strip()
        full_response_codes = seed_codes + cand["codes"]

        # Calculate aggregate score against the reference
        agg_score = evaluation_metrics.aggregate_score(full_response_text, full_response_codes, reference_text)

        # Calculate semantic score of the *entire* sequence
        sem_score = semantic_score(full_response_codes, succession_matrix)

        # Combine scores (e.g., weighted average)
        # Give slightly more weight to the aggregate score which includes BLEU/ROUGE?
        final_score = (agg_score * 0.6) + (sem_score * 0.4)

        scored_candidates.append({
            "codes": cand["codes"], # Still store only *new* codes
            "text": cand["text"], # Still store only *new* text
            "source": cand["source"],
            "agg_score": round(agg_score, 1),
            "sem_score": round(sem_score, 1),
            "final_score": round(final_score, 1)
        })
        # print(f"[Ranker Debug] Cand: '{cand['text']}' | Agg: {agg_score:.1f} | Sem: {sem_score:.1f} | Final: {final_score:.1f}")


    # Sort candidates by final_score descending
    scored_candidates.sort(key=lambda x: x["final_score"], reverse=True)

    # Return the best candidate
    # print(f"[Ranker] Best candidate: '{scored_candidates[0]['text']}' (Score: {scored_candidates[0]['final_score']:.1f})")
    return scored_candidates[0]


# --- Refactored Conversation Function ---

def _prepare_input(input_text: str, word_to_code: Dict[str, str], instance_id: str,
                     traceability_logger: TraceabilityLogger) -> Optional[List[str]]:
    """Convert input text to codes and handle unknown words."""
    codes, unknown_words = text_to_codes(input_text, word_to_code)
    if unknown_words:
        error_msg = f"Input contains unknown words: {', '.join(unknown_words)}"
        logger.info(f"[Input ERROR] {error_msg}")
        traceability_logger.log_error(instance_id, error_msg, {"input": input_text})
        return None # Reject input with unknown words
    if not codes:
         error_msg = "Input text could not be converted to any codes."
         logger.info(f"[Input ERROR] {error_msg}")
         traceability_logger.log_error(instance_id, error_msg, {"input": input_text})
         return None
    return codes

def _validate_input_codes(codes: List[str], succession_matrix: Dict[str, List[str]],
                           prob_matrix: Dict[str, Dict[str, float]], instance_id: str,
                           traceability_logger: TraceabilityLogger) -> bool:
    """Validate the initial sequence of codes."""
    is_valid, message = validate_sequence(codes, succession_matrix, prob_matrix)
    logger.info(f"[Input Validation] {message}")
    if not is_valid:
        traceability_logger.log_event(instance_id, "input_validation_failed", {"codes": codes, "reason": message})
    return is_valid

def _get_reference_text(context_memory: ContextualMemory, instance_id: str) -> str:
     """ Get the last response from memory as reference, or empty string """
     # Try loading current instance first
     context = context_memory.load_instance(instance_id)
     if context and context.get("responses"):
          # Make sure responses is a list and return last element
          responses = context["responses"]
          if isinstance(responses, list) and responses:
               return responses[-1]

     # Fallback: load most recent instance from DB if current context is empty/missing
     recent_instances = context_memory.get_recent_instances(limit=1)
     if recent_instances:
          recent_context = recent_instances[0]
          if recent_context and recent_context.get("responses"):
               responses = recent_context["responses"]
               if isinstance(responses, list) and responses:
                    return responses[-1]

     return "" # Default empty reference


def semantic_conversation(input_text: str, word_to_code: Dict[str, str], code_to_word: Dict[str, str],
                          model: GPT2LMHeadModel, tokenizer: GPT2TokenizerFast,
                          succession_matrix: Dict[str, List[str]], prob_matrix: Dict[str, Dict[str, float]],
                          grammar_codes: Dict[str, str], deductive_engine: DeductiveCognitionEngine,
                          belief_module: BeliefModule, context_memory: ContextualMemory,
                          ethical_constitution: EthicalConstitution, traceability_logger: TraceabilityLogger,
                          evaluation_metrics: EvaluationMetrics):
    """Handles a single turn of the semantic conversation."""

    instance_id = hashlib.md5(input_text.encode()).hexdigest()
    # Cycle count could be based on DB entries or just incrementing counter
    cycle_count = len(context_memory.memory) + 1 # Simple cycle count

    logger.info(f"\n--- Turn Start (Cycle {cycle_count}, Instance: {instance_id}) ---")
    logger.info(f"You: {input_text}")
    start_time = time.time()

    # 1. Prepare and Validate Input
    input_codes = _prepare_input(input_text, word_to_code, instance_id, traceability_logger)
    if input_codes is None:
        logger.info("AI: Input contains unknown words or could not be processed. Please reformulate.")
        logger.info(f"--- Turn End (Prep Failed) ---")
        return

    logger.info(f"[Input Codes] {' '.join(input_codes)}")

    if not _validate_input_codes(input_codes, succession_matrix, prob_matrix, instance_id, traceability_logger):
        logger.info("AI: Input sequence is not semantically/grammatically valid. Please reformulate.")
        logger.info(f"--- Turn End (Validation Failed) ---")
        return

    # 2. Generate and Rank Candidates
    best_candidate = None
    generated_codes = []
    reply_text = ""
    agg_score = 0.0
    sem_score = 0.0
    source = "none"

    try:
        candidates = generate_codes(
            input_codes, model, tokenizer, code_to_word, grammar_codes,
            deductive_engine, belief_module, prob_matrix, succession_matrix,
            current_cycle=cycle_count
        )

        if candidates:
            reference_text = _get_reference_text(context_memory, instance_id)
            # print(f"[DEBUG] Using reference text for ranking: '{reference_text}'")
            best_candidate = rank_candidates(candidates, evaluation_metrics, reference_text, input_codes, succession_matrix)

        if best_candidate:
             generated_codes = best_candidate["codes"] # New codes only
             reply_text = best_candidate["text"] # New text only
             agg_score = best_candidate["agg_score"]
             sem_score = best_candidate["sem_score"]
             source = best_candidate["source"]
             logger.info(f"[Generation] Best candidate (Source: {source}): '{reply_text}' (Codes: {' '.join(generated_codes)})")
        else:
             logger.info("[Generation] No valid candidates were generated.")
             reply_text = "[System: No valid response generated]" # Placeholder text

    except Exception as e:
        logger.error(f"[ERROR] Exception during candidate generation or ranking: {e}")
        traceability_logger.log_error(instance_id, f"Generation/Ranking Exception: {e}")
        reply_text = "[System: Error during generation]"

    # 3. Ethical Validation
    full_reply_text = (codes_to_text(input_codes, code_to_word) + " " + reply_text).strip() # Full text for context memory
    full_reply_codes = input_codes + generated_codes # Full codes for context memory

    is_ethical, ethical_msg = ethical_constitution.validate_decision(reply_text, generated_codes) # Validate *new* part
    logger.info(f"[Ethics Validation] {ethical_msg}")

    if not is_ethical:
        traceability_logger.log_event(instance_id, "ethical_rejection",
                                      {"reply_attempt": reply_text, "codes": generated_codes, "reason": ethical_msg})
        logger.info(f"AI: (Response rejected by ethical constitution) {ethical_msg}")
        # Optionally: Could try generating again here, but simple rejection is safer for stability
        logger.info(f"--- Turn End (Ethics Failed) ---")
        return # Stop processing if unethical

    # 4. Final Evaluation & Output Formatting (if ethical)
    # Scores (agg_score, sem_score) are already calculated during ranking
    # factual_s = evaluation_metrics.check_factual_coherence(reply_text, generated_codes) # Factual check is weak

    # Determine quality feedback based on score
    if agg_score < 50: quality_feedback = "⚠️ Response quality low."
    elif agg_score < 70: quality_feedback = "⚠️ Acceptable response, potential for improvement."
    elif agg_score < 90: quality_feedback = "✅ Good response."
    else: quality_feedback = "✅ Excellent semantic response."

    output_message = f"AI (Source: {source}, AggScore: {agg_score}%, SemScore: {sem_score}%) → {reply_text}"

    # 5. Logging and Memory Update
    log_entry = {
        "event": "conversation_turn",
        "input_text": input_text,
        "input_codes": input_codes,
        "generated_codes": generated_codes, # New codes only
        "reply_text": reply_text, # New text only
        "full_reply_text": full_reply_text, # Full text for context
        "generation_source": source,
        "semantic_score": sem_score,
        "aggregate_score": agg_score,
        # "factual_score": factual_s,
        "ethical_feedback": ethical_msg,
        "quality_feedback": quality_feedback,
        "cycle_count": cycle_count,
        "processing_time_ms": round((time.time() - start_time) * 1000)
    }
    traceability_logger.log(instance_id, log_entry)
    traceability_logger.log_code_usage(full_reply_codes, instance_id) # Log usage for the full sequence

    # Save relevant parts to contextual memory
    context_data = {
         "input": input_text,
         "codes": input_codes, # Store original input codes
         "responses": [full_reply_text], # Store the full reply generated this turn
         "scores": {"agg": agg_score, "sem": sem_score}, # Store key scores
         "pos_sequences": [c[:2] for c in full_reply_codes if is_valid_semantic_code(c)] # POS of full sequence
    }
    # If instance exists, append response, else create new
    existing_context = context_memory.load_instance(instance_id)
    if existing_context and isinstance(existing_context.get("responses"), list):
         context_data["responses"] = existing_context["responses"] + [full_reply_text]
         # Maybe average scores or store score history? For simplicity, store latest scores.

    context_memory.save_instance(instance_id, context_data)

    # 6. Print final output
    logger.info(output_message)
    logger.info(f"   {quality_feedback}")
    logger.info(f"--- Turn End (Success, Time: {log_entry['processing_time_ms']}ms) ---")


def save_model(model, tokenizer, save_directory):
    """Saves the model and tokenizer to the specified directory."""
    try:
        model.save_pretrained(save_directory)
        tokenizer.save_pretrained(save_directory)
        logger.info(f"[INFO] Model and tokenizer saved successfully to: {save_directory}")
    except Exception as e:
        logger.error(f"[ERROR] Failed to save model/tokenizer to {save_directory}: {e}")

# --- 5. Data Preparation & Training Functions (for TRAIN mode) ---

def download_gutenberg_text(url: str) -> Optional[str]:
     """Downloads text content from a Gutenberg URL."""
     try:
         logger.info(f"[Corpus] Downloading: {url}")
         response = requests.get(url, timeout=30) # Add timeout
         response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
         # Decode assuming UTF-8, handle potential errors
         txt_data = response.content.decode('utf-8', errors='ignore')
         logger.info(f"[Corpus] Downloaded {len(txt_data)} characters.")
         # Basic cleaning (remove headers/footers if possible - complex)
         # txt_data = clean_gutenberg_text(txt_data) # Requires custom cleaning logic
         return txt_data
     except requests.exceptions.RequestException as e:
         logger.error(f"[ERROR] Failed to download {url}: {e}")
         return None
     except Exception as e:
         logger.error(f"[ERROR] Unexpected error processing {url}: {e}")
         return None

def extract_wikipedia_text(bz2_file: str, output_dir: str) -> List[str]:
     """Extracts text from Wikipedia dump using WikiExtractor."""
     logger.info(f"[Corpus] Extracting Wikipedia text from {bz2_file}...")
     # Check if wikiextractor is installed? Assumed for now.
     # Command needs to be run in a subprocess
     import subprocess
     cmd = [
         sys.executable, # Use the current Python interpreter
         "-m", "wikiextractor.WikiExtractor",
         bz2_file,
         "--output", output_dir,
         "--json", # Output JSON lines
         "--bytes", "100M", # Process in chunks
         #"--processes", "4" # Use multiple processes if beneficial
     ]
     try:
          subprocess.run(cmd, check=True, capture_output=True, text=True)
          logger.info("[Corpus] WikiExtractor finished.")
     except FileNotFoundError:
          logger.error("[ERROR] WikiExtractor not found. Make sure it's installed (`pip install wikiextractor`).")
          return []
     except subprocess.CalledProcessError as e:
         logger.error(f"[ERROR] WikiExtractor failed with exit code {e.returncode}.")
         logger.info(f"Stderr: {e.stderr}")
         logger.info(f"Stdout: {e.stdout}")
         return []
     except Exception as e:
         logger.error(f"[ERROR] Unexpected error running WikiExtractor: {e}")
         return []

     # Read extracted text from JSON files
     wiki_texts = []
     extracted_files_count = 0
     try:
         for root, _, files in os.walk(output_dir):
             for file in files:
                 if file.startswith("wiki_"): # Files created by wikiextractor
                     file_path = os.path.join(root, file)
                     extracted_files_count += 1
                     with open(file_path, "r", encoding='utf-8') as f:
                         for line in f:
                             try:
                                 data = json.loads(line)
                                 wiki_texts.append(data.get("text", ""))
                             except json.JSONDecodeError:
                                 # print(f"[WARN] Skipping invalid JSON line in {file_path}")
                                 pass
         logger.info(f"[Corpus] Read {len(wiki_texts)} documents from {extracted_files_count} extracted Wikipedia files.")
     except Exception as e:
         logger.error(f"[ERROR] Failed to read extracted Wikipedia files: {e}")

     return wiki_texts


def load_and_process_corpus() -> str:
    """Loads corpus from Wikipedia and Gutenberg."""
    all_texts = []

    # --- Wikipedia ---
    # Needs manual download first, e.g., using wget outside Python or requests
    wiki_dump_file = os.path.join(CORPUS_DIR, "enwiki-latest-pages-articles1.xml-p1p41242.bz2")
    wiki_extract_dir = os.path.join(CORPUS_DIR, "wiki_extracted")

    if not os.path.exists(wiki_dump_file):
         logger.warning(f"[WARN] Wikipedia dump file not found: {wiki_dump_file}")
         logger.warning("[WARN] Skipping Wikipedia processing. Download it manually or adjust path.")
         # Example download command (run manually or in a setup script):
         # wget -P {CORPUS_DIR} https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles1.xml-p1p41242.bz2
    else:
        wiki_texts = extract_wikipedia_text(wiki_dump_file, wiki_extract_dir)
        all_texts.extend(wiki_texts)

    # --- Gutenberg ---
    gutenberg_urls = [
        "https://www.gutenberg.org/files/2701/2701-0.txt", # Moby Dick
        "https://www.gutenberg.org/files/2600/2600-0.txt", # War and Peace
        "https://www.gutenberg.org/files/1342/1342-0.txt", # Pride and Prejudice
        "https://www.gutenberg.org/files/84/84-0.txt",     # Frankenstein
        "https://www.gutenberg.org/files/11/11-0.txt",     # Alice's Adventures
        "https://www.gutenberg.org/files/1661/1661-0.txt", # Sherlock Holmes
        "https://www.gutenberg.org/files/4300/4300-0.txt"  # Ulysses
    ]
    for url in gutenberg_urls:
        gutenberg_text = download_gutenberg_text(url)
        if gutenberg_text:
            all_texts.append(gutenberg_text)

    if not all_texts:
         logger.error("[ERROR] No corpus text loaded. Cannot proceed with training preparation.")
         return ""

    combined_text = "\n\n".join(filter(None, all_texts)) # Join with double newline
    logger.info(f"[INFO] Total combined corpus size: {len(combined_text)} characters")
    return combined_text

def generate_semantic_vocabulary(text: str, max_words: int, nlp_spacy) -> List[Dict]:
    """Generates the semantic vocabulary list from text."""
    logger.info(f"[Vocab Gen] Processing text for vocabulary (limit: {max_words} words)...")
    # Basic cleaning: lowercase and remove non-alphabetic/whitespace chars
    cleaned_text = re.sub(r'[^a-z\s]', '', text.lower())

    # Process text in chunks to manage memory
    chunks = [cleaned_text[i:i+CORPUS_CHUNK_SIZE] for i in range(0, len(cleaned_text), CORPUS_CHUNK_SIZE)]
    logger.info(f"[Vocab Gen] Processing {len(chunks)} chunks...")

    all_tokens = []
    for idx, chunk in enumerate(chunks):
        if idx % 10 == 0: logger.info(f"[Vocab Gen] Processing chunk {idx+1}/{len(chunks)}")
        try:
            # Process using spaCy - disable unnecessary pipes for speed
            doc = nlp_spacy(chunk, disable=["parser", "ner"])
            all_tokens.extend([token.text for token in doc if token.is_alpha])
        except Exception as e:
            logger.error(f"[ERROR] spaCy processing failed on chunk {idx+1}: {e}")
            # Continue with next chunk?

    if not all_tokens:
        logger.error("[ERROR] No tokens extracted from corpus.")
        return []

    # Calculate word frequencies
    freq = Counter(all_tokens)
    logger.info(f"[Vocab Gen] Total unique words found: {len(freq)}")

    # Get the most common words up to the limit
    most_common_words = [word for word, count in freq.most_common(max_words)]
    logger.info(f"[Vocab Gen] Selected top {len(most_common_words)} words for vocabulary.")

    # --- Semantic Code Generation Logic (using WordNet and heuristics) ---
    # POS mapping (example, adjust as needed)
    pos_map = {"NOUN": "01", "VERB": "02", "ADJ": "03", "ADV": "04", "ADP": "05",
               "PROPN": "01", # Map Proper Nouns to Nouns
               "DET": "06", "PRON": "07", "AUX": "02", # Map Aux to Verbs
               "PART": "99", "CCONJ": "99", "SCONJ": "99", "NUM": "99", "SYM": "99", "X": "99", "INTJ":"99", "PUNCT":"99", "SPACE":"99"}

    # Subcategory mapping (using simplified Penn tags from spaCy)
    subcat_map = {"NN": "01", "NNS": "02", "NNP": "08", "NNPS": "09", # Nouns
                  "VB": "03", "VBP": "04", "VBZ": "05", "VBD": "06", "VBG": "07", "VBN": "07", # Verbs (map VBN to VBG)
                  "JJ": "10", "JJR": "11", "JJS": "12", # Adjectives
                  "RB": "13", "RBR": "14", "RBS": "15", # Adverbs
                  # Add others if needed, mapping to '00' or specific codes
                  }

    def get_wordnet_polarity(word):
         # Simple heuristic based on seed words and antonyms
         synsets = wn.synsets(word)
         if not synsets: return "0" # Neutral
         # Check first few synsets? Just first one for speed.
         synset = synsets[0]
         positive_seeds = {"good", "happy", "love", "joy", "great", "excellent", "positive", "fortunate", "correct", "active"}
         negative_seeds = {"bad", "sad", "hate", "pain", "terrible", "awful", "negative", "unfortunate", "wrong", "passive"}

         lemmas = synset.lemmas()
         for lemma in lemmas:
             lemma_name = lemma.name().lower()
             if lemma_name in positive_seeds: return "1"
             if lemma_name in negative_seeds: return "2"
             # Check antonyms
             for antonym in lemma.antonyms():
                  antonym_name = antonym.name().lower()
                  if antonym_name in positive_seeds: return "2" # Antonym of positive is negative
                  if antonym_name in negative_seeds: return "1" # Antonym of negative is positive
         return "0" # Neutral if no strong signal

    def get_wordnet_domain(word):
         # Simplified domain check using hypernyms
         synsets = wn.synsets(word)
         if not synsets: return "00" # Undefined
         # Look up hypernym chain (e.g., 3 levels)
         hypernym_names = set()
         q = deque([(synsets[0], 0)])
         visited = {synsets[0]}

         while q:
              syn, level = q.popleft()
              if level > 3: continue
              hypernym_names.add(syn.lexname()) # Use lexicographer file names (e.g., noun.animal)
              # Simplified check based on common hypernym roots
              name_part = syn.name().split('.')[0]
              if name_part in {"science", "physics", "chemistry", "biology", "math"}: return "01" # Science/Math
              if name_part in {"art", "music", "literature", "linguistics"}: return "02" # Arts/Humanities
              if name_part in {"technology", "computer", "tool", "engineering"}: return "03" # Technology
              if name_part in {"feeling", "emotion", "state", "motive"}: return "04" # Abstract/Psychological
              if name_part in {"animal", "plant", "location", "object", "substance"}: return "05" # Nature/Physical
              if name_part in {"person", "group", "social", "economy", "politics"}: return "06" # Social/People

              for hyper in syn.hypernyms():
                   if hyper not in visited:
                       q.append((hyper, level + 1))
                       visited.add(hyper)
         # Check lexnames if direct name match fails
         if any(d.startswith('noun.animal') or d.startswith('noun.plant') for d in hypernym_names): return "05"
         if any(d.startswith('noun.feeling') for d in hypernym_names): return "04"
         if any(d.startswith('noun.person') or d.startswith('noun.group') for d in hypernym_names): return "06"
         if any(d.startswith('noun.artifact') for d in hypernym_names): return "03" # Artifact -> Technology?

         return "00" # Undefined


    def get_wordnet_register(word):
         # Simple heuristic based on seed words
         formal_seeds = {"thus", "hence", "consequently", "furthermore", "albeit", "shall", "whom", "reside", "endeavor"}
         informal_seeds = {"gonna", "wanna", "ain't", "dude", "cool", "yeah", "nope", "stuff", "hey"}
         if word in formal_seeds: return "1"
         if word in informal_seeds: return "2"
         # Maybe check synset definitions for formality clues? (Complex)
         return "0" # Neutral

    def get_abstract_code(word, synsets):
        # Heuristic: check hypernyms for abstract concepts
        if not synsets: return "0" # Concrete default
        abstract_roots = {'abstraction', 'psychological_feature', 'attribute', 'state', 'measure', 'relation'}
        q = deque([(synsets[0], 0)])
        visited = {synsets[0]}
        while q:
             syn, level = q.popleft()
             if level > 5: continue # Limit search depth
             if syn.name().split('.')[0] in abstract_roots: return "1"
             # Check definition for keywords? (Less reliable)
             # if any(w in syn.definition() for w in ["idea", "concept", "quality"]): return "1"
             for hyper in syn.hypernyms():
                  if hyper not in visited:
                      q.append((hyper, level + 1))
                      visited.add(hyper)
        return "0"

    def get_rhyme_code(word):
        # Simple rhyme code based on last 2 chars (as before)
        if len(word) < 2: return "00"
        try:
             # Ensure result is always two digits
             return str(sum(ord(c) for c in word[-2:]) % 100).zfill(2)
        except:
             return "00" # Fallback

    def get_local_id(word):
        # Consistent local ID based on hash (as before)
        try:
            # Use sha1 for shorter hash, take modulo 10
            hash_val = int(hashlib.sha1(word.encode()).hexdigest(), 16)
            return str(hash_val % 10)
        except:
            return "0" # Fallback

    # --- Generate records ---
    records = []
    processed_count = 0
    for word in most_common_words:
        if not word: continue # Skip empty strings if any

        # Process with spaCy to get POS and Tag
        try:
             doc = nlp_spacy(word)
             token = doc[0] if doc else None
        except Exception as e:
             logger.info(f"[Vocab Gen WARN] spaCy failed for word '{word}': {e}")
             token = None

        # Get WordNet synsets
        synsets = wn.synsets(word)

        # --- Construct the 12-digit code ---
        pos_c = pos_map.get(token.pos_, "99") if token else "99"
        subcat_c = subcat_map.get(token.tag_, "00") if token else "00"
        polarity_c = get_wordnet_polarity(word)
        abstract_c = get_abstract_code(word, synsets)
        domain_c = get_wordnet_domain(word)
        register_c = get_wordnet_register(word)
        rhyme_c = get_rhyme_code(word)
        local_id_c = get_local_id(word)

        # Combine codes - ensure 12 digits
        # Format: POS(2) + SubCat(2) + Polarity(1) + Abstract(1) + Domain(2) + Register(1) + Rhyme(2) + LocalID(1) = 12
        code = f"{pos_c}{subcat_c}{polarity_c}{abstract_c}{domain_c}{register_c}{rhyme_c}{local_id_c}"

        # Final validation - should be exactly 12 digits
        if len(code) != SEMANTIC_CODE_LENGTH or not code.isdigit():
             logger.info(f"[Vocab Gen WARN] Generated invalid code '{code}' for word '{word}'. Skipping.")
             continue # Skip word if code is malformed

        records.append({"word": word, "semantic_code": code})
        processed_count += 1
        if processed_count % 5000 == 0:
             logger.info(f"[Vocab Gen] Generated codes for {processed_count}/{len(most_common_words)} words...")

    logger.info(f"[Vocab Gen] Finished generating {len(records)} vocabulary records.")
    return records


def build_probabilistic_succession_matrix(corpus: str, pos_map: Dict[str, str], nlp_spacy) -> Dict[str, Dict[str, float]]:
    """Builds probabilistic POS succession matrix from corpus text."""
    logger.info("[Prob Matrix] Building probabilistic POS succession matrix...")
    # Limit corpus size for practical speed
    corpus_limit = 2000000 # Process first 2M chars
    text_to_process = corpus[:corpus_limit]

    transitions = Counter() # Stores (pos1, pos2) counts
    pos_totals = Counter() # Stores total counts for each pos1

    logger.info(f"[Prob Matrix] Processing {len(text_to_process)} characters...")
    # Process text in chunks to manage memory
    chunk_size = 500000
    text_chunks = [text_to_process[i:i+chunk_size] for i in range(0, len(text_to_process), chunk_size)]

    processed_tokens = 0
    for idx, chunk in enumerate(text_chunks):
        # print(f"[Prob Matrix] Processing chunk {idx+1}/{len(text_chunks)}")
        try:
            # Disable parser and NER for speed when only POS is needed
            doc = nlp_spacy(chunk, disable=["parser", "ner"])
            current_sentence_pos = []
            for token in doc:
                # Check sentence boundary
                if token.is_sent_start and current_sentence_pos:
                    # Process the completed sentence
                    for i in range(len(current_sentence_pos) - 1):
                        pos1 = current_sentence_pos[i]
                        pos2 = current_sentence_pos[i+1]
                        transitions[(pos1, pos2)] += 1
                        pos_totals[pos1] += 1
                        processed_tokens += 1
                    current_sentence_pos = [] # Reset for new sentence

                # Add token's POS if it's alphabetic
                if token.is_alpha:
                    pos_code = pos_map.get(token.pos_, "99") # Use the same POS map as vocab gen
                    current_sentence_pos.append(pos_code)

            # Process the last sentence in the chunk
            if current_sentence_pos:
                for i in range(len(current_sentence_pos) - 1):
                    pos1 = current_sentence_pos[i]
                    pos2 = current_sentence_pos[i+1]
                    transitions[(pos1, pos2)] += 1
                    pos_totals[pos1] += 1
                    processed_tokens += 1

        except Exception as e:
            logger.error(f"[ERROR] spaCy processing failed during prob matrix build on chunk {idx+1}: {e}")

    logger.info(f"[Prob Matrix] Processed approximately {processed_tokens} tokens.")

    # Calculate probabilities
    prob_matrix: Dict[str, Dict[str, float]] = {}
    for (pos1, pos2), count in transitions.items():
        total_count = pos_totals.get(pos1)
        if total_count and total_count > 0:
            if pos1 not in prob_matrix:
                prob_matrix[pos1] = {}
            prob_matrix[pos1][pos2] = count / total_count
        # else: # Handle cases where pos1 might not have transitions (optional)
        #      print(f"[Prob Matrix WARN] POS tag '{pos1}' found in transitions but not in totals.")

    logger.info(f"[Prob Matrix] Built matrix with {len(prob_matrix)} source POS tags.")
    return prob_matrix


def prepare_training_data(text: str, word_to_code: Dict[str, str], output_file: str, nlp_spacy):
     """Prepares the augmented training data file (word|code)."""
     logger.info(f"[Training Prep] Preparing training data for {output_file}...")

     # Limit text size if necessary
     text_limit = 5000000 # Process 5M chars for training data
     text_to_process = text[:text_limit]

     processed_sentences = 0
     unknown_word_count = 0
     total_word_count = 0

     # Process in chunks
     chunk_size = 500000
     text_chunks = [text_to_process[i:i+chunk_size] for i in range(0, len(text_to_process), chunk_size)]

     with open(output_file, "w", encoding="utf-8") as f:
          for idx, chunk in enumerate(text_chunks):
               logger.info(f"[Training Prep] Processing chunk {idx+1}/{len(text_chunks)}")
               try:
                    doc = nlp_spacy(chunk, disable=["ner"]) # Keep parser for sentence boundaries
                    for sent in doc.sents:
                         augmented_sentence = []
                         for token in sent:
                              if token.is_alpha:
                                   total_word_count += 1
                                   code = word_to_code.get(token.text.lower())
                                   if code: # Only include words with valid codes
                                       augmented_sentence.append(f"{token.text}|{code}")
                                   else:
                                        unknown_word_count += 1

                         if augmented_sentence: # Write sentence only if it has coded words
                              f.write(" ".join(augmented_sentence) + "\n")
                              processed_sentences += 1
               except Exception as e:
                    logger.error(f"[ERROR] Failed processing chunk {idx+1} for training data: {e}")

     logger.info(f"[Training Prep] Finished preparing training data.")
     logger.info(f"[Training Prep] Processed {processed_sentences} sentences.")
     if total_word_count > 0:
          unknown_ratio = unknown_word_count / total_word_count
          logger.info(f"[Training Prep] Encountered {unknown_word_count} unknown words ({unknown_ratio:.2%} of total).")
     logger.info(f"[Training Prep] Augmented corpus saved to: {output_file}")


# --- 6. Main Execution Logic ---

if __name__ == "__main__":

    # --- Load Core NLP Model (needed in both modes) ---
    try:
        logger.info("[INFO] Loading spaCy model (en_core_web_sm)...")
        nlp_spacy = spacy.load("en_core_web_sm")
        logger.info("[INFO] spaCy model loaded.")
    except OSError:
        logger.error("[ERROR] spaCy model 'en_core_web_sm' not found.")
        logger.error("[ERROR] Please run: python -m spacy download en_core_web_sm")
        sys.exit(1)
    except Exception as e:
         logger.error(f"[ERROR] Unexpected error loading spaCy model: {e}")
         sys.exit(1)

    # ==============================
    # ---       TRAIN MODE       ---
    # ==============================
    if EXECUTION_MODE == "TRAIN":
        logger.info("\n--- Starting Execution in TRAIN Mode ---")

        # 1. Load and Process Corpus
        logger.info("\n[TRAIN] === Step 1: Loading Corpus ===")
        combined_text = load_and_process_corpus()
        if not combined_text:
            logger.error("[ERROR] Corpus loading failed. Exiting train mode.")
            sys.exit(1)

        # 2. Generate Semantic Vocabulary
        logger.info("\n[TRAIN] === Step 2: Generating Semantic Vocabulary ===")
        # Example POS map needed for vocab generation - ensure consistency with matrix building
        _pos_map_for_vocab = {"NOUN": "01", "VERB": "02", "ADJ": "03", "ADV": "04", "ADP": "05",
                              "PROPN": "01", "DET": "06", "PRON": "07", "AUX": "02",
                              "PART": "99", "CCONJ": "99", "SCONJ": "99", "NUM": "99", "SYM": "99", "X": "99", "INTJ":"99", "PUNCT":"99", "SPACE":"99"} # Reuse later
        vocab_records = generate_semantic_vocabulary(combined_text, MAX_VOCAB_WORDS, nlp_spacy)
        if not vocab_records:
             logger.error("[ERROR] Semantic vocabulary generation failed. Exiting train mode.")
             sys.exit(1)
        try:
            with open(SEMANTIC_VOCAB_FILE, "w", encoding="utf-8") as f:
                json.dump(vocab_records, f, indent=2)
            logger.info(f"[INFO] Semantic vocabulary saved to: {SEMANTIC_VOCAB_FILE}")
            # Load generated vocab immediately for next steps
            word_to_code, code_to_word, _ = load_semantic_vocab(SEMANTIC_VOCAB_FILE)
            if not word_to_code or not code_to_word: raise ValueError("Failed to load newly generated vocab")
        except Exception as e:
            logger.error(f"[ERROR] Failed to save or reload semantic vocabulary: {e}")
            sys.exit(1)

        # 3. Generate GSL Vocabulary (Assumed User Provides File)
        logger.info("\n[TRAIN] === Step 3: Generating GSL Vocabulary ===")
        logger.info(f"[INFO] Skipping GSL vocabulary generation. Expecting file: {GRAMMAR_VOCAB_FILE}")
        # Placeholder for the missing 'generate_gsl_vocabulary' function call if it existed
        # generate_gsl_vocabulary(vocab_records, GRAMMAR_VOCAB_FILE)
        # Verify the provided file exists
        if not os.path.exists(GRAMMAR_VOCAB_FILE):
             logger.warning(f"[WARN] GSL Vocabulary file '{GRAMMAR_VOCAB_FILE}' not found. Grammar checks will fail later.")


        # 4. Build Probabilistic Succession Matrix
        logger.info("\n[TRAIN] === Step 4: Building Probabilistic Succession Matrix ===")
        # Ensure POS map is defined
        pos_map_for_matrix = _pos_map_for_vocab # Use the same map
        prob_matrix = build_probabilistic_succession_matrix(combined_text, pos_map_for_matrix, nlp_spacy)
        if not prob_matrix:
             logger.error("[ERROR] Failed to build probabilistic matrix. Exiting train mode.")
             sys.exit(1)
        try:
            with open(PROB_MATRIX_FILE, "w", encoding="utf-8") as f:
                json.dump(prob_matrix, f, indent=2)
            logger.info(f"[INFO] Probabilistic succession matrix saved to: {PROB_MATRIX_FILE}")
        except Exception as e:
            logger.error(f"[ERROR] Failed to save probabilistic matrix: {e}")
            sys.exit(1)

        # 5. Prepare Training Data
        logger.info("\n[TRAIN] === Step 5: Preparing Training Data ===")
        prepare_training_data(combined_text, word_to_code, TRAINING_CORPUS_FILE, nlp_spacy)
        if not os.path.exists(TRAINING_CORPUS_FILE) or os.path.getsize(TRAINING_CORPUS_FILE) == 0:
             logger.error("[ERROR] Training data preparation failed or produced empty file. Exiting train mode.")
             sys.exit(1)

        # 6. Fine-tune GPT-2 Model
        logger.info("\n[TRAIN] === Step 6: Fine-tuning GPT-2 Model ===")
        try:
            logger.info(f"[INFO] Loading base model {MODEL_NAME} and tokenizer...")
            tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
            # Set padding token - crucial for batching
            if tokenizer.pad_token is None:
                 tokenizer.pad_token = tokenizer.eos_token
                 logger.info("[INFO] Set tokenizer pad_token to eos_token.")

            model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
            # Move model to GPU if available
            model.to(GPU_DEVICE)
            logger.info(f"[INFO] Model loaded to {GPU_DEVICE}.")

            logger.info(f"[INFO] Loading training dataset from: {TRAINING_CORPUS_FILE}")
            # Load dataset from the text file
            # Use 'text' column name as expected by transformers
            dataset = load_dataset('text', data_files={'train': TRAINING_CORPUS_FILE})['train']

            def tokenize_function(examples):
                # Tokenize text, handle truncation and padding later via collator
                return tokenizer(examples["text"], truncation=False) # Don't truncate here

            logger.info("[INFO] Tokenizing dataset...")
            tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
            # Filter out potentially empty sequences after tokenization if needed
            # tokenized_dataset = tokenized_dataset.filter(lambda example: len(example['input_ids']) > 0)
            logger.info(f"[INFO] Tokenization complete. Dataset size: {len(tokenized_dataset)}")

            # Data Collator for Language Modeling (handles padding)
            data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

            # Training Arguments
            training_args = TrainingArguments(
                output_dir=FINETUNED_MODEL_PATH,
                overwrite_output_dir=True,
                num_train_epochs=TRAINING_EPOCHS,
                per_device_train_batch_size=TRAINING_BATCH_SIZE,
                save_steps=500,      # Save checkpoint every 500 steps
                save_total_limit=2,  # Keep only the last 2 checkpoints
                logging_steps=100,   # Log metrics every 100 steps
                prediction_loss_only=True, # Only compute loss during eval (if eval set existed)
                fp16=FP16_TRAINING if GPU_DEVICE == "cuda" else False, # Use mixed precision if on GPU
                # gradient_accumulation_steps=2, # Optional: if batch size needs to be effectively larger
                learning_rate=5e-5, # Default learning rate
                # Add other arguments as needed (e.g., weight_decay)
            )

            # Initialize Trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                data_collator=data_collator,
                train_dataset=tokenized_dataset,
                # eval_dataset=... # Add tokenized validation set if available
            )

            # Start Training
            logger.info("[INFO] Starting model training...")
            train_result = trainer.train()
            logger.info("[INFO] Training finished.")

            # Log training metrics
            metrics = train_result.metrics
            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)

            # Save final model
            logger.info("[INFO] Saving final trained model...")
            save_model(model, tokenizer, FINETUNED_MODEL_PATH)

        except Exception as e:
             logger.error(f"[ERROR] Exception during model training: {e}")
             # Consider more specific exception handling (e.g., CUDA OOM)
             sys.exit(1)

        logger.info("\n--- TRAIN Mode Execution Finished Successfully ---")


    # ==============================
    # ---    INFERENCE MODE      ---
    # ==============================
    elif EXECUTION_MODE == "INFERENCE":
        logger.info("\n--- Starting Execution in INFERENCE Mode ---")

        # 1. Load Vocabularies and Matrices
        logger.info("\n[INFERENCE] === Step 1: Loading Vocabularies & Matrices ===")
        word_to_code, code_to_word, _ = load_semantic_vocab(SEMANTIC_VOCAB_FILE)
        grammar_codes = load_grammar_vocab(GRAMMAR_VOCAB_FILE)
        prob_matrix = load_json_file(PROB_MATRIX_FILE, "Probabilistic Matrix")

        if not word_to_code or not code_to_word or not grammar_codes or not prob_matrix:
             logger.error("[ERROR] Failed to load necessary vocabulary or matrix files. Exiting inference mode.")
             sys.exit(1)

        # Define fixed succession matrix (can also be loaded from file if preferred)
        succession_matrix = {
            "01": ["01", "02", "03", "04", "05", "06", "07", "99"], # Noun -> Noun, Verb, Adj, Adv, Prep, Det, Pron, Other
            "02": ["01", "03", "04", "05", "06", "07", "99"], # Verb -> Noun, Adj, Adv, Prep, Det, Pron, Other
            "03": ["01", "03", "99"],                         # Adj -> Noun, Adj, Other
            "04": ["02", "03", "04", "05", "99"],             # Adv -> Verb, Adj, Adv, Prep, Other
            "05": ["01", "03", "06", "07", "99"],             # Prep -> Noun, Adj, Det, Pron, Other
            "06": ["01", "03", "99"],                         # Det -> Noun, Adj, Other
            "07": ["01", "02", "05", "99"],                   # Pron -> Noun, Verb, Prep, Other
            "99": ["01", "02", "03", "04", "05", "06", "07", "99"]  # Other -> Any
        }

        # 2. Load Fine-tuned Model
        logger.info("\n[INFERENCE] === Step 2: Loading Fine-tuned Model ===")
        try:
            if not os.path.exists(FINETUNED_MODEL_PATH):
                 raise FileNotFoundError(f"Fine-tuned model directory not found: {FINETUNED_MODEL_PATH}")

            logger.info(f"[INFO] Loading fine-tuned model and tokenizer from: {FINETUNED_MODEL_PATH}")
            tokenizer = GPT2TokenizerFast.from_pretrained(FINETUNED_MODEL_PATH)
            model = GPT2LMHeadModel.from_pretrained(FINETUNED_MODEL_PATH)

            # Ensure padding token is set
            if tokenizer.pad_token is None:
                 tokenizer.pad_token = tokenizer.eos_token
                 logger.info("[INFO] Set tokenizer pad_token to eos_token.")

            # Move model to GPU if available
            model.to(GPU_DEVICE)
            model.eval() # Set model to evaluation mode
            logger.info(f"[INFO] Fine-tuned model loaded successfully to {GPU_DEVICE}.")

        except FileNotFoundError as e:
             logger.error(f"[ERROR] {e}")
             logger.error("[ERROR] Please ensure the model has been trained (run in TRAIN mode) or the path is correct.")
             sys.exit(1)
        except Exception as e:
            logger.error(f"[ERROR] Failed to load fine-tuned model: {e}")
            sys.exit(1)

        # 3. Initialize Epistemic Modules & Logger
        logger.info("\n[INFERENCE] === Step 3: Initializing Modules ===")
        # Define axioms (keep hardcoded or load from config file)
        axioms = {
            "cats run in places": "places include forests", # A -> B
            "places include forests": "in forest",           # B -> C (Allows 'cats run' -> 'in forest')
            "love brings emotions": "emotions include peace",
            "emotions include peace": "and peace",
            "formal speech requires clarity": "clarity requires wisdom",
            "clarity requires wisdom": "with wisdom",
            "sad dog walks slowly": "slowly indicates low energy", # Example
            "low energy requires rest": "needs rest"              # Example
        }

        try:
            deductive_engine = DeductiveCognitionEngine(axioms, word_to_code, succession_matrix)
            belief_module = BeliefModule(decay_rate=BELIEF_DECAY_RATE, max_beliefs=MAX_BELIEFS)
            belief_module.set_word_to_code(word_to_code) # Critical: provide vocab to belief module
            context_memory = ContextualMemory(DB_PATH)
            ethical_constitution = EthicalConstitution(succession_matrix, word_to_code)
            traceability_logger = TraceabilityLogger(TRACEABILITY_LOG_PATH)
            evaluation_metrics = EvaluationMetrics(succession_matrix, axioms, word_to_code) # Pass word_to_code
            logger.info("[INFO] All modules initialized successfully.")
        except Exception as e:
             logger.error(f"[ERROR] Failed to initialize modules: {e}")
             sys.exit(1)

        # 4. Start Conversation Loop (Example Usage)
        logger.info("\n[INFERENCE] === Step 4: Starting Interaction ===")
        logger.info("Enter text to start conversation (or type 'quit' to exit):")

        # Example test texts from original code
        test_texts = [
             "The happy cat runs quickly",
             "Cat run tree", # Likely invalid input due to unknown words/grammar
             "Love brings joy",
             "The sad dog walks slowly",
             "Sir therefore speak clearly", # Requires 'sir', 'therefore' in vocab
             "xyz", # Invalid input
             "run fast now"
        ]

        # --- Run predefined tests ---
        logger.info("\n--- Running Predefined Tests ---")
        for text in test_texts:
             if not text or text.lower() == 'quit': continue
             semantic_conversation(
                 text, word_to_code, code_to_word, model, tokenizer,
                 succession_matrix, prob_matrix, grammar_codes,
                 deductive_engine, belief_module, context_memory,
                 ethical_constitution, traceability_logger, evaluation_metrics
             )
             time.sleep(0.5) # Pause slightly between turns

        # --- Interactive Loop ---
        logger.info("\n--- Starting Interactive Loop (type 'quit' to exit) ---")
        while True:
            try:
                user_input = input("You: ")
            except EOFError: # Handle pipe closing etc.
                 logger.info("\n[INFO] Input stream closed. Exiting.")
                 break

            if user_input.lower() == 'quit':
                break
            if not user_input.strip():
                 continue

            semantic_conversation(
                 user_input, word_to_code, code_to_word, model, tokenizer,
                 succession_matrix, prob_matrix, grammar_codes,
                 deductive_engine, belief_module, context_memory,
                 ethical_constitution, traceability_logger, evaluation_metrics
            )

        logger.info("\n--- INFERENCE Mode Execution Finished ---")

    else:
        logger.error(f"[ERROR] Unknown execution mode: {EXECUTION_MODE}")
        parser.print_help()
        sys.exit(1)
