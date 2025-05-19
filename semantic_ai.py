==
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

print(f"[INFO] Running in {EXECUTION_MODE} mode.")
print(f"[INFO] Using device: {GPU_DEVICE}")
print(f"[INFO] Base directory: {BASE_DIR}")

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
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset, load_dataset # Modified for handling text file directly
from nltk.corpus import wordnet as wn
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import random
import numpy as np
from typing import Dict, List, Optional, Callable, Tuple, Any

# --- Download NLTK data if not present ---
try:
    nltk.data.find('corpora/wordnet.zip')
except nltk.downloader.DownloadError:
    print("[INFO] Downloading NLTK wordnet data...")
    nltk.download('wordnet', quiet=True)
try:
    nltk.data.find('corpora/omw-1.4.zip')
except nltk.downloader.DownloadError:
    print("[INFO] Downloading NLTK omw-1.4 data...")
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
         print("[POS Check] Warning: Sequence contains invalid or non-semantic codes.")
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
        print(f"[ERROR] {description} file not found at: {file_path}")
        return None
    try:
        with open(file_path, "r", encoding='utf-8') as f:
            data = json.load(f)
        print(f"[INFO] Successfully loaded {description} from: {file_path}")
        return data
    except json.JSONDecodeError as e:
        print(f"[ERROR] Failed to decode JSON from {description} file: {file_path}. Error: {e}")
        return None
    except Exception as e:
        print(f"[ERROR] Unexpected error loading {description} file: {file_path}. Error: {e}")
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
         print(f"[WARN] Found {malformed_entries} malformed entries in semantic vocabulary file.")

    if not word_to_code:
         print("[ERROR] No valid entries found in the semantic vocabulary.")
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
         print(f"[WARN] Found {malformed_entries} malformed entries in grammar vocabulary file.")

    if not grammar_codes:
        print("[ERROR] No valid entries found in the grammar vocabulary.")
        return None

    return grammar_codes

def text_to_codes(text: str, word_to_code: Dict[str, str]) -> Tuple[List[str], List[str]]:
    """Converts text to a list of semantic codes, returning codes and unknown words."""
    if not isinstance(text, str):
        print("[ERROR] Input text is not a string.")
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

class DeductiveCognitionEngine:
    def __init__(self, axioms: Dict[str, str], word_to_code: Dict[str, str], succession_matrix: Dict[str, List[str]]):
        """Initialize deductive engine with axioms and POS succession."""
        self.axioms = axioms if axioms else {}
        self.word_to_code = word_to_code if word_to_code else {}
        self.succession_matrix = succession_matrix if succession_matrix else {}
        self.chain: List[str] = []
        self._extend_axioms_from_matrix()

    def _extend_axioms_from_matrix(self):
        """Extend axioms using POS succession rules."""
        if not self.succession_matrix: return
        for pos1, allowed_pos2_list in self.succession_matrix.items():
            if not pos1 or not isinstance(allowed_pos2_list, list): continue
            for pos2 in allowed_pos2_list:
                if pos2: # Add rule only if both POS are valid
                    self.axioms[f"pos_{pos1}_follows"] = f"pos_{pos2}_next"

    def _match_rule(self, premise: str, rule_premise: str, codes: List[str]) -> bool:
        """Check premise match (case-insensitive) and POS sequence validity."""
        if not isinstance(premise, str) or not isinstance(rule_premise, str):
             return False
        # Basic premise matching (can be enhanced with NLP techniques if needed)
        premise_match = premise.lower() in rule_premise.lower()
        # Check POS validity of the *codes* associated with the input premise
        pos_valid = check_full_pos_sequence(codes, self.succession_matrix)
        return premise_match and pos_valid

    def _get_code(self, consequence_word: str) -> str:
        """Get the semantic code for a consequence word."""
        return self.word_to_code.get(consequence_word.lower(), "N/A")

    def _single_deduction(self, current_premise: str, current_codes: List[str]) -> Optional[Dict]:
        """Performs one step of deduction."""
        for rule_premise, consequence in self.axioms.items():
            if self._match_rule(current_premise, rule_premise, current_codes):
                # Consequence might be a single word or a phrase
                # We need a robust way to get the code for the *main* part of the consequence
                # Simplification: Assume consequence is often a single word or target state
                consequence_word = consequence.split()[-1] # Heuristic: take last word
                code = self._get_code(consequence_word)
                # Return even if code is N/A, let the caller handle it
                return {"text": consequence, "code": code}
        return None

    def _chain_deductions(self, initial_premise: str, initial_codes: List[str], max_hops: int = DEDUCTION_MAX_HOPS) -> Optional[Dict]:
        """Support multi-hop deductions. Returns the *first* valid deduction found in the chain."""
        current_premise = initial_premise
        current_codes = initial_codes[:] # Use a copy
        self.chain = [] # Reset chain for this deduction attempt

        for hop in range(max_hops):
            deduction = self._single_deduction(current_premise, current_codes)

            if not deduction:
                # print(f"[Deduction] Chain stopped at hop {hop+1}: No rule matched '{current_premise}'")
                break # Stop if no rule matches

            consequence_text = deduction["text"]
            consequence_code = deduction["code"]

            self.chain.append(f"Hop {hop+1}: {current_premise} |- {consequence_text} (Code: {consequence_code})")

            # Check if the deduced consequence is valid and usable
            if consequence_code != "N/A":
                # Found a valid deduction in the chain
                return {"text": consequence_text, "code": consequence_code}
            else:
                # Deduction resulted in N/A code, continue chain with the text?
                # Or stop? Current logic stops if code is N/A. Let's allow chaining based on text.
                print(f"[Deduction] Warning: Hop {hop+1} consequence '{consequence_text}' has N/A code. Continuing chain.")
                current_premise = consequence_text # Use the text as the next premise
                # How to get codes for the new text premise? This is tricky.
                # Simplification: Continue chain without updating codes, relying only on text match.
                # Or: Stop if code is N/A to ensure codes remain grounded. Let's stop.
                print(f"[Deduction] Stopping chain at hop {hop+1} due to N/A code.")
                break # Stop chain if code is N/A

        # print(f"[Deduction] Chain finished after {len(self.chain)} hops. No valid code deduction found.")
        return None # No valid deduction found within max_hops

    def deduce(self, premise: str, codes: List[str]) -> Optional[Dict]:
        """Deduce a conclusion using inference chains, with fallback to POS succession."""
        if not premise or not codes or not self.word_to_code or not self.succession_matrix:
             print("[Deduction] Error: Engine not properly initialized or invalid input.")
             return None

        # 1. Try chained deduction
        deduction = self._chain_deductions(premise, codes)
        if deduction: # Already checked for N/A code inside _chain_deductions
            print(f"[Deduction] Success via chain: {self.chain[-1]}")
            return deduction

        # 2. Fallback: Use POS succession matrix if deduction fails
        last_valid_code = next((c for c in reversed(codes) if is_valid_semantic_code(c)), None)

        if last_valid_code:
            last_pos = last_valid_code[:2]
            allowed_next_pos = self.succession_matrix.get(last_pos, [])
            if allowed_next_pos:
                # Find candidate words with allowed next POS
                candidates = [
                    (word, code) for word, code in self.word_to_code.items()
                    if is_valid_semantic_code(code) and code[:2] in allowed_next_pos
                ]
                if candidates:
                    chosen_word, chosen_code = random.choice(candidates)
                    fallback_deduction = {"text": chosen_word, "code": chosen_code}
                    self.chain.append(f"Fallback (POS): {premise} (last POS: {last_pos}) |- {chosen_word} (Code: {chosen_code})")
                    print(f"[Deduction] Success via POS fallback: {self.chain[-1]}")
                    return fallback_deduction

        # print(f"[Deduction] Failed for premise: '{premise}'")
        return None

class BeliefModule:
    def __init__(self, decay_rate: float = BELIEF_DECAY_RATE, max_beliefs: int = MAX_BELIEFS):
        """Initialize belief module with pseudo-Bayesian metric."""
        self.beliefs: Dict[str, Dict] = {} # Key: proposition text, Value: {"B": float, "cycles": int, "codes": List[str]}
        self.decay_rate = decay_rate
        self.max_beliefs = max_beliefs
        self.word_to_code: Optional[Dict[str, str]] = None # Set via method

    def set_word_to_code(self, word_to_code: Dict[str, str]):
        """Set the vocabulary for code lookups."""
        self.word_to_code = word_to_code

    def _get_proposition_code(self, proposition: str) -> str:
        """Get the semantic code for a single-word proposition."""
        # Assumes proposition is often a single concept/word for simplicity
        # More complex propositions would need parsing.
        if not self.word_to_code: return "N/A"
        # Simple heuristic: use the code of the last word if multiple words
        prop_word = proposition.split()[-1] if proposition else ""
        return self.word_to_code.get(prop_word.lower(), "N/A")

    def _calculate_support(self, proposition_code: str, context_codes: List[str]) -> float:
        """Calculate support based on POS compatibility of the proposition with the context."""
        if not is_valid_semantic_code(proposition_code) or not context_codes:
            return 0.0 # No support if prop code invalid or no context

        last_context_code = next((c for c in reversed(context_codes) if is_valid_semantic_code(c)), None)
        if not last_context_code:
            return 0.1 # Minimal support if context has no valid codes

        # Simple heuristic: Higher support if proposition's POS is grammatically likely after context's last POS
        # This requires the succession matrix, which isn't directly available here.
        # Simplified heuristic: Check if POS matches the last context POS (less accurate)
        # proposition_pos = proposition_code[:2]
        # last_context_pos = last_context_code[:2]
        # return 0.8 if proposition_pos == last_context_pos else 0.2
        # Using a fixed moderate support for simplicity without succession matrix:
        return 0.5 # Neutral support value

    def _calculate_noise(self, proposition_code: str) -> float:
        """Calculate noise based on code validity."""
        # Higher noise if the proposition doesn't map to a valid code
        return 0.1 if is_valid_semantic_code(proposition_code) else 0.7 # Increased noise for N/A

    def _prune_beliefs(self):
        """Remove weakest beliefs if exceeding max_beliefs."""
        if len(self.beliefs) > self.max_beliefs:
            # Sort beliefs by strength (ascending)
            sorted_beliefs = sorted(self.beliefs.items(), key=lambda item: item[1]['B'])
            # Remove the weakest ones until count is within limit
            num_to_remove = len(self.beliefs) - self.max_beliefs
            for i in range(num_to_remove):
                 prop_to_remove = sorted_beliefs[i][0]
                 # print(f"[Belief Pruning] Removing weak belief: '{prop_to_remove}' (B={self.beliefs[prop_to_remove]['B']:.2f})")
                 del self.beliefs[prop_to_remove]

    def evaluate_belief(self, proposition: str, context_codes: List[str], current_cycle: int) -> float:
        """Evaluate belief strength with cycle-based decay."""
        if not proposition or not self.word_to_code:
             return 0.0

        prop_code = self._get_proposition_code(proposition)

        if proposition not in self.beliefs:
            # Initialize new belief with neutral strength
            self.beliefs[proposition] = {"B": 0.5, "cycles": current_cycle, "codes": [prop_code]} # Store prop code
            # Check if adding this exceeds limit
            if len(self.beliefs) > self.max_beliefs:
                 self._prune_beliefs() # Prune before calculating for the new one
                 # Re-check if the new one was immediately pruned (unlikely with B=0.5 unless limit is very small)
                 if proposition not in self.beliefs:
                      # print(f"[Belief] New belief '{proposition}' pruned immediately.")
                      return 0.0

        belief_data = self.beliefs[proposition]
        elapsed_cycles = max(0, current_cycle - belief_data["cycles"])

        # Recalculate support and noise based on current context
        S = self._calculate_support(prop_code, context_codes)
        N = self._calculate_noise(prop_code)

        # Belief update formula (example): B_new = B_old * (1 - decay)^elapsed * (S - N)
        # Using formula from original code: max(0.0, 1 - S - N) * (1 - self.decay_rate) ** elapsed
        # This seems to decay towards an equilibrium based on S and N, not multiplicative decay of old value.
        # Let's use the original formula for consistency:
        current_belief_strength = max(0.0, 1.0 - S - N)
        decay_factor = (1.0 - self.decay_rate) ** elapsed_cycles
        new_belief_strength = current_belief_strength * decay_factor

        # Update belief state
        belief_data["B"] = new_belief_strength
        belief_data["cycles"] = current_cycle
        # Update codes associated with belief? Maybe store context codes too?
        # Original stored 'codes', maybe meaning context? Let's store prop_code.
        belief_data["codes"] = [prop_code] if prop_code != "N/A" else []

        # Prune if belief becomes too weak
        if belief_data["B"] < 0.01: # Threshold for pruning weak beliefs
            # print(f"[Belief Pruning] Belief '{proposition}' dropped below threshold (B={belief_data['B']:.3f}). Removing.")
            del self.beliefs[proposition]
            return 0.0 # Return 0 as it was removed

        return belief_data["B"]

    def update_belief(self, proposition: str, evidence_strength: float, context_codes: List[str], current_cycle: int):
        """Update belief based on external evidence."""
        if not proposition or not self.word_to_code: return

        prop_code = self._get_proposition_code(proposition)
        # Clamp evidence strength between 0 and 1
        clamped_evidence = max(0.0, min(1.0, evidence_strength))

        self.beliefs[proposition] = {
            "B": clamped_evidence,
            "cycles": current_cycle,
            "codes": [prop_code] if prop_code != "N/A" else []
        }
        # Prune if necessary after update
        self._prune_beliefs()
        # print(f"[Belief Update] Belief '{proposition}' set to B={clamped_evidence:.2f}")

    def get_best_belief(self) -> Optional[Dict]:
        """Get the proposition with the highest belief strength."""
        if not self.beliefs:
            return None

        # Find the proposition with the maximum 'B' value
        try:
            # Use default=None for robustness if beliefs becomes empty during iteration (unlikely)
             best_item = max(self.beliefs.items(), key=lambda item: item[1].get('B', 0.0), default=None)
        except ValueError: # Handles case where beliefs might be empty
             return None

        if best_item is None:
             return None

        best_proposition, best_data = best_item
        best_code = best_data.get("codes", ["N/A"])[0] if best_data.get("codes") else "N/A" # Get stored code

        return {
            "proposition": best_proposition,
            "code": best_code, # Return the code associated with the belief itself
            "B": best_data.get('B', 0.0)
        }

class ContextualMemory:
    def __init__(self, db_path: str):
        """Initialize contextual memory with SQLite for persistence."""
        self.db_path = db_path
        self.memory: Dict[str, Dict] = {} # In-memory cache
        self.conn: Optional[sqlite3.Connection] = None
        self.cursor: Optional[sqlite3.Cursor] = None
        self._init_db()

    def _init_db(self):
        """Initialize SQLite connection and create table if needed."""
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False) # Allow multi-thread access if needed
            self.cursor = self.conn.cursor()
            self.cursor.execute('''CREATE TABLE IF NOT EXISTS instances
                                   (id TEXT PRIMARY KEY, data TEXT, timestamp REAL)''')
            self.conn.commit()
            print(f"[INFO] Contextual Memory DB initialized at: {self.db_path}")
        except sqlite3.Error as e:
            print(f"[ERROR] Failed to initialize SQLite DB at {self.db_path}: {e}")
            self.conn = None
            self.cursor = None

    def save_instance(self, instance_id: str, data: Dict):
        """Save instance to cache and SQLite."""
        if not instance_id or not data:
            print("[ERROR] Attempted to save instance with invalid ID or data.")
            return

        # Add timestamp to data before saving
        data_with_timestamp = data.copy()
        data_with_timestamp['timestamp'] = time.time()

        # Save to in-memory cache
        self.memory[instance_id] = data_with_timestamp

        # Save to SQLite if connection is available
        if self.conn and self.cursor:
            try:
                # Serialize data to JSON string
                data_json = json.dumps(data_with_timestamp)
                self.cursor.execute("INSERT OR REPLACE INTO instances (id, data, timestamp) VALUES (?, ?, ?)",
                                    (instance_id, data_json, data_with_timestamp['timestamp']))
                self.conn.commit()
            except json.JSONDecodeError as e:
                 print(f"[ERROR] Could not serialize instance data to JSON for ID {instance_id}. Data: {data}. Error: {e}")
            except sqlite3.Error as e:
                print(f"[ERROR] SQLite error saving instance {instance_id}: {e}")
                # Consider rollback or other error handling
            except Exception as e:
                 print(f"[ERROR] Unexpected error saving instance {instance_id} to DB: {e}")
        else:
             print("[WARN] SQLite connection not available. Instance saved to memory only.")

    def load_instance(self, instance_id: str) -> Optional[Dict]:
        """Load instance from cache or SQLite."""
        if not instance_id: return None

        # Check cache first
        if instance_id in self.memory:
            return self.memory[instance_id]

        # Load from SQLite if connection is available
        if self.conn and self.cursor:
            try:
                self.cursor.execute("SELECT data FROM instances WHERE id = ?", (instance_id,))
                result = self.cursor.fetchone()
                if result and result[0]:
                    # Deserialize data from JSON string
                    data = json.loads(result[0])
                    # Store in cache
                    self.memory[instance_id] = data
                    return data
            except json.JSONDecodeError as e:
                 print(f"[ERROR] Could not deserialize instance data from DB for ID {instance_id}. Error: {e}")
            except sqlite3.Error as e:
                print(f"[ERROR] SQLite error loading instance {instance_id}: {e}")
            except Exception as e:
                 print(f"[ERROR] Unexpected error loading instance {instance_id} from DB: {e}")

        return None # Not found in cache or DB, or error occurred

    def merge_instances(self, instance_ids: List[str]) -> Dict:
        """Merge data from multiple instances for emergent context."""
        merged_data = {"codes": [], "responses": [], "pos_sequences": [], "merged_ids": []}
        if not instance_ids:
             return merged_data

        for iid in instance_ids:
            instance_data = self.load_instance(iid)
            if instance_data:
                merged_data["codes"].extend(instance_data.get("codes", []))
                merged_data["responses"].extend(instance_data.get("responses", []))
                # Generate POS sequences if not stored directly
                pos_seq = [c[:2] for c in instance_data.get("codes", []) if is_valid_semantic_code(c)]
                merged_data["pos_sequences"].extend(pos_seq)
                merged_data["merged_ids"].append(iid)
        return merged_data

    def get_recent_instances(self, limit: int = 5) -> List[Dict]:
         """ Retrieves the most recent instances from the database """
         if not self.conn or not self.cursor:
             print("[WARN] DB not available for retrieving recent instances.")
             # Fallback: try sorting in-memory cache? (less reliable)
             return []

         recent_instances = []
         try:
             self.cursor.execute("SELECT data FROM instances ORDER BY timestamp DESC LIMIT ?", (limit,))
             results = self.cursor.fetchall()
             for row in results:
                 if row and row[0]:
                     try:
                         data = json.loads(row[0])
                         recent_instances.append(data)
                     except json.JSONDecodeError as e:
                          print(f"[ERROR] Failed to decode recent instance data from DB: {e}")
         except sqlite3.Error as e:
             print(f"[ERROR] SQLite error retrieving recent instances: {e}")
         except Exception as e:
             print(f"[ERROR] Unexpected error retrieving recent instances: {e}")

         return recent_instances


    def __del__(self):
        """Close the database connection upon object deletion."""
        if self.conn:
            try:
                self.conn.close()
                print("[INFO] Contextual Memory DB connection closed.")
            except sqlite3.Error as e:
                print(f"[ERROR] Error closing SQLite connection: {e}")

class EthicalConstitution:
    def __init__(self, succession_matrix: Dict[str, List[str]], word_to_code: Dict[str, str]):
        """Initialize ethical constitution with refined rules."""
        self.word_to_code = word_to_code if word_to_code else {}
        self.succession_matrix = succession_matrix if succession_matrix else {}
        # Define rules as named functions for clarity
        self.rules: List[Callable[[str, List[str]], Tuple[bool, str]]] = [
            self._rule_repetitiveness,
            self._rule_semantic_score,
            self._rule_pos_sequence,
            self._rule_transparency,
            # self._rule_reversibility, # Reversibility check was complex and potentially unreliable, disable for now
        ]
        self.feedback_history = deque(maxlen=5) # Store last 5 validated responses {decision, approved}

    def _rule_repetitiveness(self, response: str, codes: List[str]) -> Tuple[bool, str]:
        """Rule 1: Check if response is too similar to recent approved responses."""
        if not response: return True, "Repetitiveness: OK (empty response)"
        response_words = set(response.lower().split())
        if not response_words: return True, "Repetitiveness: OK (no words)"

        for history_item in self.feedback_history:
            if history_item.get("approved"):
                hist_words = set(history_item.get("decision", "").lower().split())
                if not hist_words: continue

                # Calculate Jaccard similarity
                intersection = len(response_words.intersection(hist_words))
                union = len(response_words.union(hist_words))
                overlap = intersection / union if union > 0 else 0.0

                # Threshold for excessive overlap (e.g., 60% similar)
                if overlap > 0.6:
                    msg = f"Repetitiveness: Failed (Overlap {overlap:.2f} with previous: '{history_item['decision']}')"
                    # print(f"[Ethics] {msg}") # Debug
                    return False, msg
        return True, "Repetitiveness: OK"

    def _rule_semantic_score(self, response: str, codes: List[str]) -> Tuple[bool, str]:
        """Rule 2: Check if the semantic score of the generated codes is above a threshold."""
        if not codes: return True, "Semantic Score: OK (no codes)" # Or False? Assume OK if no codes generated.

        score = semantic_score(codes, self.succession_matrix)
        threshold = 50.0 # Minimum acceptable semantic score

        if score >= threshold:
            return True, f"Semantic Score: OK (Score={score:.1f} >= {threshold})"
        else:
            msg = f"Semantic Score: Failed (Score={score:.1f} < {threshold})"
            # print(f"[Ethics] {msg}") # Debug
            return False, msg

    def _rule_pos_sequence(self, response: str, codes: List[str]) -> Tuple[bool, str]:
        """Rule 3: Verify POS sequence compatibility using the utility function."""
        if not codes: return True, "POS Sequence: OK (no codes)"

        is_valid = check_full_pos_sequence(codes, self.succession_matrix)
        if is_valid:
            return True, "POS Sequence: OK"
        else:
            msg = "POS Sequence: Failed (Invalid POS transition detected)"
            # print(f"[Ethics] {msg}") # Debug
            return False, msg

    def _rule_transparency(self, response: str, codes: List[str]) -> Tuple[bool, str]:
        """Rule 4: Check if response is explainable (all words are in vocabulary)."""
        if not response: return True, "Transparency: OK (empty response)"
        if not self.word_to_code:
             print("[Ethics Transparency] Warning: word_to_code map not available.")
             return False, "Transparency: Failed (Vocabulary unavailable)"

        response_words = response.lower().split()
        unknown_words = [word for word in response_words if word not in self.word_to_code]

        if not unknown_words:
            return True, "Transparency: OK"
        else:
            msg = f"Transparency: Failed (Unknown words: {', '.join(unknown_words)})"
            # print(f"[Ethics] {msg}") # Debug
            return False, msg

    # def _rule_reversibility(self, response: str, codes: List[str]) -> Tuple[bool, str]:
    #     """(Disabled) Rule 5: Check if the opposite response is syntactically valid."""
    #     # This rule is complex and its value is debatable. Disabled for stability.
    #     return True, "Reversibility: Skipped"

    def validate_decision(self, decision: str, codes: List[str]) -> Tuple[bool, str]:
        """Validate the decision against all defined ethical rules."""
        violated_rules_feedback = []
        for i, rule_func in enumerate(self.rules):
            try:
                is_valid, feedback_msg = rule_func(decision, codes)
                if not is_valid:
                    violated_rules_feedback.append(f"Rule {i+1}: {feedback_msg}")
            except Exception as e:
                print(f"[ERROR] Exception during ethical rule {i+1} execution: {e}")
                violated_rules_feedback.append(f"Rule {i+1}: Failed due to internal error")

        if violated_rules_feedback:
            full_feedback = "Decision Rejected -> " + "; ".join(violated_rules_feedback)
            self.feedback_history.append({"decision": decision, "feedback": full_feedback, "approved": False})
            return False, full_feedback
        else:
            final_msg = "Decision Approved."
            self.feedback_history.append({"decision": decision, "feedback": final_msg, "approved": True})
            return True, final_msg

class TraceabilityLogger:
    def __init__(self, log_path: str):
        """Initialize logger for traceability, saving logs as JSON lines."""
        self.log_path_base = log_path
        self.memory_logs: Dict[str, List[Dict]] = {} # Optional in-memory cache
        os.makedirs(log_path, exist_ok=True)
        print(f"[INFO] Traceability Logger initialized. Logs will be saved in: {log_path}")

    def _get_log_file_path(self, instance_id: str) -> str:
        """Constructs the path for the instance's log file."""
        # Use instance_id for filename, sanitize if needed
        safe_instance_id = re.sub(r'[^a-zA-Z0-9_\-]', '_', instance_id)
        return os.path.join(self.log_path_base, f"{safe_instance_id}_log.jsonl")

    def log(self, instance_id: str, entry: Dict):
        """Logs an entry for a given instance ID to a JSON Lines file."""
        if not instance_id or not entry:
             print("[Logger ERROR] Attempted to log with invalid instance ID or empty entry.")
             return

        # Add timestamp to every log entry
        entry_with_timestamp = entry.copy()
        entry_with_timestamp['timestamp'] = time.time()
        entry_with_timestamp['instance_id'] = instance_id # Ensure instance ID is in the log record

        # --- Optional: In-memory logging ---
        # if instance_id not in self.memory_logs:
        #     self.memory_logs[instance_id] = []
        # self.memory_logs[instance_id].append(entry_with_timestamp)
        # Limit memory usage if using cache:
        # if len(self.memory_logs[instance_id]) > 100: self.memory_logs[instance_id].pop(0)

        # --- File Logging (JSON Lines format) ---
        file_path = self._get_log_file_path(instance_id)
        try:
            log_line = json.dumps(entry_with_timestamp)
            with open(file_path, "a", encoding='utf-8') as f: # Append mode
                f.write(log_line + "\n")
        except TypeError as e:
            print(f"[Logger ERROR] Failed to serialize log entry to JSON for instance {instance_id}. Entry: {entry_with_timestamp}. Error: {e}")
        except IOError as e:
            print(f"[Logger ERROR] Failed to write to log file {file_path}. Error: {e}")
        except Exception as e:
             print(f"[Logger ERROR] Unexpected error writing log for instance {instance_id}: {e}")

    def log_event(self, instance_id: str, event_type: str, data: Dict):
         """Logs a structured event."""
         self.log(instance_id, {"event": event_type, "data": data})

    def log_error(self, instance_id: str, error_msg: str, context: Optional[Dict] = None):
         """Logs an error event."""
         log_entry = {"event": "error", "message": error_msg}
         if context:
             log_entry["context"] = context
         self.log(instance_id, log_entry)

    def log_hallucination(self, instance_id: str, response: str, issue: str):
        """Logs a detected potential hallucination."""
        self.log_event(instance_id, "hallucination_detected", {"response": response, "issue": issue})

    def log_code_usage(self, codes: List[str], instance_id: str):
        """Log usage statistics of semantic code dimensions."""
        if not codes: return
        valid_codes = [c for c in codes if is_valid_semantic_code(c)]
        if not valid_codes: return

        usage = {
            "pos": Counter(c[:2] for c in valid_codes),
            "subcat": Counter(c[2:4] for c in valid_codes),
            "polarity": Counter(c[4] for c in valid_codes),
            "abstract": Counter(c[5] for c in valid_codes),
            "domain": Counter(c[6:8] for c in valid_codes),
            "register": Counter(c[8] for c in valid_codes),
            "rhyme": Counter(c[9:11] for c in valid_codes), # Assuming rhyme is c[9:11]
            "local_id": Counter(c[11] for c in valid_codes) # Assuming local ID is c[11]
        }
        # Convert Counter objects to plain dicts for JSON serialization
        usage_serializable = {k: dict(v) for k, v in usage.items()}
        self.log_event(instance_id, "code_usage_stats", {"stats": usage_serializable, "code_count": len(valid_codes)})


class EvaluationMetrics:
    # Using BLEU smoothing function
    chencherry = SmoothingFunction()

    def __init__(self, succession_matrix: Dict[str, List[str]], axioms: Dict[str, str], word_to_code: Optional[Dict[str, str]] = None):
        """Initialize metrics with ROUGE and factual coherence."""
        self.succession_matrix = succession_matrix if succession_matrix else {}
        self.axioms = axioms if axioms else {}
        self.word_to_code = word_to_code if word_to_code else {} # Needed for factual coherence check
        self.cache: Dict[str, float] = {} # Cache for aggregate scores
        self.scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    @staticmethod
    def lexical_diversity(text: str) -> float:
        """Calculate Type-Token Ratio (TTR)."""
        if not isinstance(text, str) or not text.strip(): return 0.0
        words = text.lower().split()
        if not words: return 0.0
        return len(set(words)) / len(words)

    @staticmethod
    def bleu_score(reference: str, candidate: str) -> float:
        """Calculate BLEU score with smoothing."""
        if not isinstance(reference, str) or not reference.strip() or \
           not isinstance(candidate, str) or not candidate.strip():
            return 0.0

        ref_tokens = [reference.lower().split()] # List of reference token lists
        cand_tokens = candidate.lower().split()

        if not ref_tokens[0] or not cand_tokens:
             return 0.0

        try:
            # Using smoothing function 4 (Chen & Cherry) handles short sentences better
            return sentence_bleu(ref_tokens, cand_tokens, smoothing_function=EvaluationMetrics.chencherry.method4)
        except ZeroDivisionError:
             # print("[BLEU WARN] ZeroDivisionError calculating BLEU.")
             return 0.0
        except Exception as e:
            print(f"[BLEU ERROR] Unexpected error: {e}")
            return 0.0

    def rouge_l_score(self, reference: str, candidate: str) -> float:
        """Calculate ROUGE-L F-measure."""
        if not isinstance(reference, str) or not isinstance(candidate, str):
            return 0.0

        # Handle empty strings gracefully
        if not reference.strip() or not candidate.strip():
             # Decide score for empty strings: 0 if one is non-empty, 1 if both empty?
             return 1.0 if not reference.strip() and not candidate.strip() else 0.0

        try:
            scores = self.scorer.score(reference, candidate)
            return scores['rougeL'].fmeasure
        except Exception as e:
             print(f"[ROUGE ERROR] Unexpected error: {e}")
             return 0.0

    def logical_consistency(self, codes: List[str]) -> float:
        """Calculate consistency based on POS succession rules."""
        valid_codes = [c for c in codes if is_valid_semantic_code(c)]
        if len(valid_codes) < 2:
            return 1.0 # Consider single code or empty sequence consistent

        num_pairs = len(valid_codes) - 1
        valid_transitions = 0
        for i in range(num_pairs):
            pos1 = valid_codes[i][:2]
            pos2 = valid_codes[i+1][:2]
            if is_valid_pos_succession(pos1, pos2, self.succession_matrix):
                valid_transitions += 1

        return valid_transitions / num_pairs

    def check_factual_coherence(self, response: str, response_codes: List[str]) -> float:
        """Check if the response aligns with simple axioms. Basic check."""
        if not response or not self.axioms or not self.word_to_code:
            return 0.5 # Neutral score if axioms/vocab unavailable or response empty

        # This check is very basic. It looks if the *exact response* appears as a *consequence*
        # in any axiom where the *premise* somehow matches the input codes (via text?).
        # This needs improvement for real factual checking.

        # Attempt to reconstruct a premise text from codes (simplistic)
        premise_text = codes_to_text(response_codes, self.word_to_code) if self.word_to_code else ""

        response_lower = response.lower()

        # Check if the response is a direct consequence of any axiom
        if response_lower in [c.lower() for c in self.axioms.values()]:
             return 1.0 # High score if it matches a known consequence explicitly

        # Check if the response follows logically from the *premise_text* based on axioms
        # Requires the deductive engine or similar logic here. This is complex.
        # Simplified: Check if response matches consequence IF premise_text matches rule_premise
        # for rule_premise, consequence in self.axioms.items():
        #      if premise_text and premise_text.lower() in rule_premise.lower() and response_lower == consequence.lower():
        #           return 0.9 # Reasonably high score if linked via axiom

        # Default score if no direct match found
        return 0.5

    def aggregate_score(self, response: str, codes: List[str], reference: str) -> float:
        """Calculate an aggregate quality score. Uses caching."""
        if not isinstance(response, str): response = ""
        if not isinstance(reference, str): reference = ""

        # Use a tuple of codes for the cache key component
        codes_tuple = tuple(codes)
        cache_key = f"{response}_{reference}_{codes_tuple}"

        if cache_key in self.cache:
            return self.cache[cache_key]

        # Calculate individual metrics
        bleu = self.bleu_score(reference, response)
        rouge = self.rouge_l_score(reference, response)
        diversity = self.lexical_diversity(response)
        logic = self.logical_consistency(codes)
        # factual = self.check_factual_coherence(response, codes) # Factual check is weak, maybe lower weight
        semantic = semantic_score(codes, self.succession_matrix) / 100.0 # Normalize to 0-1

        # Define weights (these may need tuning)
        w_bleu = 0.20
        w_rouge = 0.20
        w_diversity = 0.10 # Lower weight as it can be easily gamed
        w_logic = 0.25 # Higher weight for internal consistency
        w_semantic = 0.25 # Higher weight for semantic consistency
        # w_factual = 0.10

        # Calculate weighted average, ensuring total weight is 1.0
        score = (bleu * w_bleu +
                 rouge * w_rouge +
                 diversity * w_diversity +
                 logic * w_logic +
                 semantic * w_semantic) # +
                 # factual * w_factual)

        final_score = round(score * 100, 1) # Scale to 0-100

        self.cache[cache_key] = final_score
        # Optional: Limit cache size
        if len(self.cache) > 10000:
             try:
                 self.cache.pop(next(iter(self.cache)))
             except StopIteration:
                 pass

        return final_score

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
                print(f"[ERROR] GPT-2 generation failed: {e}")
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
        print(f"[Input ERROR] {error_msg}")
        traceability_logger.log_error(instance_id, error_msg, {"input": input_text})
        return None # Reject input with unknown words
    if not codes:
         error_msg = "Input text could not be converted to any codes."
         print(f"[Input ERROR] {error_msg}")
         traceability_logger.log_error(instance_id, error_msg, {"input": input_text})
         return None
    return codes

def _validate_input_codes(codes: List[str], succession_matrix: Dict[str, List[str]],
                           prob_matrix: Dict[str, Dict[str, float]], instance_id: str,
                           traceability_logger: TraceabilityLogger) -> bool:
    """Validate the initial sequence of codes."""
    is_valid, message = validate_sequence(codes, succession_matrix, prob_matrix)
    print(f"[Input Validation] {message}")
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

    print(f"\n--- Turn Start (Cycle {cycle_count}, Instance: {instance_id}) ---")
    print(f"You: {input_text}")
    start_time = time.time()

    # 1. Prepare and Validate Input
    input_codes = _prepare_input(input_text, word_to_code, instance_id, traceability_logger)
    if input_codes is None:
        print("AI: Input contains unknown words or could not be processed. Please reformulate.")
        print(f"--- Turn End (Prep Failed) ---")
        return

    print(f"[Input Codes] {' '.join(input_codes)}")

    if not _validate_input_codes(input_codes, succession_matrix, prob_matrix, instance_id, traceability_logger):
        print("AI: Input sequence is not semantically/grammatically valid. Please reformulate.")
        print(f"--- Turn End (Validation Failed) ---")
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
             print(f"[Generation] Best candidate (Source: {source}): '{reply_text}' (Codes: {' '.join(generated_codes)})")
        else:
             print("[Generation] No valid candidates were generated.")
             reply_text = "[System: No valid response generated]" # Placeholder text

    except Exception as e:
        print(f"[ERROR] Exception during candidate generation or ranking: {e}")
        traceability_logger.log_error(instance_id, f"Generation/Ranking Exception: {e}")
        reply_text = "[System: Error during generation]"

    # 3. Ethical Validation
    full_reply_text = (codes_to_text(input_codes, code_to_word) + " " + reply_text).strip() # Full text for context memory
    full_reply_codes = input_codes + generated_codes # Full codes for context memory

    is_ethical, ethical_msg = ethical_constitution.validate_decision(reply_text, generated_codes) # Validate *new* part
    print(f"[Ethics Validation] {ethical_msg}")

    if not is_ethical:
        traceability_logger.log_event(instance_id, "ethical_rejection",
                                      {"reply_attempt": reply_text, "codes": generated_codes, "reason": ethical_msg})
        print(f"AI: (Response rejected by ethical constitution) {ethical_msg}")
        # Optionally: Could try generating again here, but simple rejection is safer for stability
        print(f"--- Turn End (Ethics Failed) ---")
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
    print(output_message)
    print(f"   {quality_feedback}")
    print(f"--- Turn End (Success, Time: {log_entry['processing_time_ms']}ms) ---")


def save_model(model, tokenizer, save_directory):
    """Saves the model and tokenizer to the specified directory."""
    try:
        model.save_pretrained(save_directory)
        tokenizer.save_pretrained(save_directory)
        print(f"[INFO] Model and tokenizer saved successfully to: {save_directory}")
    except Exception as e:
        print(f"[ERROR] Failed to save model/tokenizer to {save_directory}: {e}")

# --- 5. Data Preparation & Training Functions (for TRAIN mode) ---

def download_gutenberg_text(url: str) -> Optional[str]:
     """Downloads text content from a Gutenberg URL."""
     try:
         print(f"[Corpus] Downloading: {url}")
         response = requests.get(url, timeout=30) # Add timeout
         response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
         # Decode assuming UTF-8, handle potential errors
         txt_data = response.content.decode('utf-8', errors='ignore')
         print(f"[Corpus] Downloaded {len(txt_data)} characters.")
         # Basic cleaning (remove headers/footers if possible - complex)
         # txt_data = clean_gutenberg_text(txt_data) # Requires custom cleaning logic
         return txt_data
     except requests.exceptions.RequestException as e:
         print(f"[ERROR] Failed to download {url}: {e}")
         return None
     except Exception as e:
         print(f"[ERROR] Unexpected error processing {url}: {e}")
         return None

def extract_wikipedia_text(bz2_file: str, output_dir: str) -> List[str]:
     """Extracts text from Wikipedia dump using WikiExtractor."""
     print(f"[Corpus] Extracting Wikipedia text from {bz2_file}...")
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
          print("[Corpus] WikiExtractor finished.")
     except FileNotFoundError:
          print("[ERROR] WikiExtractor not found. Make sure it's installed (`pip install wikiextractor`).")
          return []
     except subprocess.CalledProcessError as e:
         print(f"[ERROR] WikiExtractor failed with exit code {e.returncode}.")
         print(f"Stderr: {e.stderr}")
         print(f"Stdout: {e.stdout}")
         return []
     except Exception as e:
         print(f"[ERROR] Unexpected error running WikiExtractor: {e}")
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
         print(f"[Corpus] Read {len(wiki_texts)} documents from {extracted_files_count} extracted Wikipedia files.")
     except Exception as e:
         print(f"[ERROR] Failed to read extracted Wikipedia files: {e}")

     return wiki_texts


def load_and_process_corpus() -> str:
    """Loads corpus from Wikipedia and Gutenberg."""
    all_texts = []

    # --- Wikipedia ---
    # Needs manual download first, e.g., using wget outside Python or requests
    wiki_dump_file = os.path.join(CORPUS_DIR, "enwiki-latest-pages-articles1.xml-p1p41242.bz2")
    wiki_extract_dir = os.path.join(CORPUS_DIR, "wiki_extracted")

    if not os.path.exists(wiki_dump_file):
         print(f"[WARN] Wikipedia dump file not found: {wiki_dump_file}")
         print("[WARN] Skipping Wikipedia processing. Download it manually or adjust path.")
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
         print("[ERROR] No corpus text loaded. Cannot proceed with training preparation.")
         return ""

    combined_text = "\n\n".join(filter(None, all_texts)) # Join with double newline
    print(f"[INFO] Total combined corpus size: {len(combined_text)} characters")
    return combined_text

def generate_semantic_vocabulary(text: str, max_words: int, nlp_spacy) -> List[Dict]:
    """Generates the semantic vocabulary list from text."""
    print(f"[Vocab Gen] Processing text for vocabulary (limit: {max_words} words)...")
    # Basic cleaning: lowercase and remove non-alphabetic/whitespace chars
    cleaned_text = re.sub(r'[^a-z\s]', '', text.lower())

    # Process text in chunks to manage memory
    chunks = [cleaned_text[i:i+CORPUS_CHUNK_SIZE] for i in range(0, len(cleaned_text), CORPUS_CHUNK_SIZE)]
    print(f"[Vocab Gen] Processing {len(chunks)} chunks...")

    all_tokens = []
    for idx, chunk in enumerate(chunks):
        if idx % 10 == 0: print(f"[Vocab Gen] Processing chunk {idx+1}/{len(chunks)}")
        try:
            # Process using spaCy - disable unnecessary pipes for speed
            doc = nlp_spacy(chunk, disable=["parser", "ner"])
            all_tokens.extend([token.text for token in doc if token.is_alpha])
        except Exception as e:
            print(f"[ERROR] spaCy processing failed on chunk {idx+1}: {e}")
            # Continue with next chunk?

    if not all_tokens:
        print("[ERROR] No tokens extracted from corpus.")
        return []

    # Calculate word frequencies
    freq = Counter(all_tokens)
    print(f"[Vocab Gen] Total unique words found: {len(freq)}")

    # Get the most common words up to the limit
    most_common_words = [word for word, count in freq.most_common(max_words)]
    print(f"[Vocab Gen] Selected top {len(most_common_words)} words for vocabulary.")

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
             print(f"[Vocab Gen WARN] spaCy failed for word '{word}': {e}")
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
             print(f"[Vocab Gen WARN] Generated invalid code '{code}' for word '{word}'. Skipping.")
             continue # Skip word if code is malformed

        records.append({"word": word, "semantic_code": code})
        processed_count += 1
        if processed_count % 5000 == 0:
             print(f"[Vocab Gen] Generated codes for {processed_count}/{len(most_common_words)} words...")

    print(f"[Vocab Gen] Finished generating {len(records)} vocabulary records.")
    return records


def build_probabilistic_succession_matrix(corpus: str, pos_map: Dict[str, str], nlp_spacy) -> Dict[str, Dict[str, float]]:
    """Builds probabilistic POS succession matrix from corpus text."""
    print("[Prob Matrix] Building probabilistic POS succession matrix...")
    # Limit corpus size for practical speed
    corpus_limit = 2000000 # Process first 2M chars
    text_to_process = corpus[:corpus_limit]

    transitions = Counter() # Stores (pos1, pos2) counts
    pos_totals = Counter() # Stores total counts for each pos1

    print(f"[Prob Matrix] Processing {len(text_to_process)} characters...")
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
            print(f"[ERROR] spaCy processing failed during prob matrix build on chunk {idx+1}: {e}")

    print(f"[Prob Matrix] Processed approximately {processed_tokens} tokens.")

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

    print(f"[Prob Matrix] Built matrix with {len(prob_matrix)} source POS tags.")
    return prob_matrix


def prepare_training_data(text: str, word_to_code: Dict[str, str], output_file: str, nlp_spacy):
     """Prepares the augmented training data file (word|code)."""
     print(f"[Training Prep] Preparing training data for {output_file}...")

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
               print(f"[Training Prep] Processing chunk {idx+1}/{len(text_chunks)}")
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
                    print(f"[ERROR] Failed processing chunk {idx+1} for training data: {e}")

     print(f"[Training Prep] Finished preparing training data.")
     print(f"[Training Prep] Processed {processed_sentences} sentences.")
     if total_word_count > 0:
          unknown_ratio = unknown_word_count / total_word_count
          print(f"[Training Prep] Encountered {unknown_word_count} unknown words ({unknown_ratio:.2%} of total).")
     print(f"[Training Prep] Augmented corpus saved to: {output_file}")


# --- 6. Main Execution Logic ---

if __name__ == "__main__":

    # --- Load Core NLP Model (needed in both modes) ---
    try:
        print("[INFO] Loading spaCy model (en_core_web_sm)...")
        nlp_spacy = spacy.load("en_core_web_sm")
        print("[INFO] spaCy model loaded.")
    except OSError:
        print("[ERROR] spaCy model 'en_core_web_sm' not found.")
        print("[ERROR] Please run: python -m spacy download en_core_web_sm")
        sys.exit(1)
    except Exception as e:
         print(f"[ERROR] Unexpected error loading spaCy model: {e}")
         sys.exit(1)

    # ==============================
    # ---       TRAIN MODE       ---
    # ==============================
    if EXECUTION_MODE == "TRAIN":
        print("\n--- Starting Execution in TRAIN Mode ---")

        # 1. Load and Process Corpus
        print("\n[TRAIN] === Step 1: Loading Corpus ===")
        combined_text = load_and_process_corpus()
        if not combined_text:
            print("[ERROR] Corpus loading failed. Exiting train mode.")
            sys.exit(1)

        # 2. Generate Semantic Vocabulary
        print("\n[TRAIN] === Step 2: Generating Semantic Vocabulary ===")
        # Example POS map needed for vocab generation - ensure consistency with matrix building
        _pos_map_for_vocab = {"NOUN": "01", "VERB": "02", "ADJ": "03", "ADV": "04", "ADP": "05",
                              "PROPN": "01", "DET": "06", "PRON": "07", "AUX": "02",
                              "PART": "99", "CCONJ": "99", "SCONJ": "99", "NUM": "99", "SYM": "99", "X": "99", "INTJ":"99", "PUNCT":"99", "SPACE":"99"} # Reuse later
        vocab_records = generate_semantic_vocabulary(combined_text, MAX_VOCAB_WORDS, nlp_spacy)
        if not vocab_records:
             print("[ERROR] Semantic vocabulary generation failed. Exiting train mode.")
             sys.exit(1)
        try:
            with open(SEMANTIC_VOCAB_FILE, "w", encoding="utf-8") as f:
                json.dump(vocab_records, f, indent=2)
            print(f"[INFO] Semantic vocabulary saved to: {SEMANTIC_VOCAB_FILE}")
            # Load generated vocab immediately for next steps
            word_to_code, code_to_word, _ = load_semantic_vocab(SEMANTIC_VOCAB_FILE)
            if not word_to_code or not code_to_word: raise ValueError("Failed to load newly generated vocab")
        except Exception as e:
            print(f"[ERROR] Failed to save or reload semantic vocabulary: {e}")
            sys.exit(1)

        # 3. Generate GSL Vocabulary (Assumed User Provides File)
        print("\n[TRAIN] === Step 3: Generating GSL Vocabulary ===")
        print(f"[INFO] Skipping GSL vocabulary generation. Expecting file: {GRAMMAR_VOCAB_FILE}")
        # Placeholder for the missing 'generate_gsl_vocabulary' function call if it existed
        # generate_gsl_vocabulary(vocab_records, GRAMMAR_VOCAB_FILE)
        # Verify the provided file exists
        if not os.path.exists(GRAMMAR_VOCAB_FILE):
             print(f"[WARN] GSL Vocabulary file '{GRAMMAR_VOCAB_FILE}' not found. Grammar checks will fail later.")


        # 4. Build Probabilistic Succession Matrix
        print("\n[TRAIN] === Step 4: Building Probabilistic Succession Matrix ===")
        # Ensure POS map is defined
        pos_map_for_matrix = _pos_map_for_vocab # Use the same map
        prob_matrix = build_probabilistic_succession_matrix(combined_text, pos_map_for_matrix, nlp_spacy)
        if not prob_matrix:
             print("[ERROR] Failed to build probabilistic matrix. Exiting train mode.")
             sys.exit(1)
        try:
            with open(PROB_MATRIX_FILE, "w", encoding="utf-8") as f:
                json.dump(prob_matrix, f, indent=2)
            print(f"[INFO] Probabilistic succession matrix saved to: {PROB_MATRIX_FILE}")
        except Exception as e:
            print(f"[ERROR] Failed to save probabilistic matrix: {e}")
            sys.exit(1)

        # 5. Prepare Training Data
        print("\n[TRAIN] === Step 5: Preparing Training Data ===")
        prepare_training_data(combined_text, word_to_code, TRAINING_CORPUS_FILE, nlp_spacy)
        if not os.path.exists(TRAINING_CORPUS_FILE) or os.path.getsize(TRAINING_CORPUS_FILE) == 0:
             print("[ERROR] Training data preparation failed or produced empty file. Exiting train mode.")
             sys.exit(1)

        # 6. Fine-tune GPT-2 Model
        print("\n[TRAIN] === Step 6: Fine-tuning GPT-2 Model ===")
        try:
            print(f"[INFO] Loading base model {MODEL_NAME} and tokenizer...")
            tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
            # Set padding token - crucial for batching
            if tokenizer.pad_token is None:
                 tokenizer.pad_token = tokenizer.eos_token
                 print("[INFO] Set tokenizer pad_token to eos_token.")

            model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
            # Move model to GPU if available
            model.to(GPU_DEVICE)
            print(f"[INFO] Model loaded to {GPU_DEVICE}.")

            print(f"[INFO] Loading training dataset from: {TRAINING_CORPUS_FILE}")
            # Load dataset from the text file
            # Use 'text' column name as expected by transformers
            dataset = load_dataset('text', data_files={'train': TRAINING_CORPUS_FILE})['train']

            def tokenize_function(examples):
                # Tokenize text, handle truncation and padding later via collator
                return tokenizer(examples["text"], truncation=False) # Don't truncate here

            print("[INFO] Tokenizing dataset...")
            tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
            # Filter out potentially empty sequences after tokenization if needed
            # tokenized_dataset = tokenized_dataset.filter(lambda example: len(example['input_ids']) > 0)
            print(f"[INFO] Tokenization complete. Dataset size: {len(tokenized_dataset)}")

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
            print("[INFO] Starting model training...")
            train_result = trainer.train()
            print("[INFO] Training finished.")

            # Log training metrics
            metrics = train_result.metrics
            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)

            # Save final model
            print("[INFO] Saving final trained model...")
            save_model(model, tokenizer, FINETUNED_MODEL_PATH)

        except Exception as e:
             print(f"[ERROR] Exception during model training: {e}")
             # Consider more specific exception handling (e.g., CUDA OOM)
             sys.exit(1)

        print("\n--- TRAIN Mode Execution Finished Successfully ---")


    # ==============================
    # ---    INFERENCE MODE      ---
    # ==============================
    elif EXECUTION_MODE == "INFERENCE":
        print("\n--- Starting Execution in INFERENCE Mode ---")

        # 1. Load Vocabularies and Matrices
        print("\n[INFERENCE] === Step 1: Loading Vocabularies & Matrices ===")
        word_to_code, code_to_word, _ = load_semantic_vocab(SEMANTIC_VOCAB_FILE)
        grammar_codes = load_grammar_vocab(GRAMMAR_VOCAB_FILE)
        prob_matrix = load_json_file(PROB_MATRIX_FILE, "Probabilistic Matrix")

        if not word_to_code or not code_to_word or not grammar_codes or not prob_matrix:
             print("[ERROR] Failed to load necessary vocabulary or matrix files. Exiting inference mode.")
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
        print("\n[INFERENCE] === Step 2: Loading Fine-tuned Model ===")
        try:
            if not os.path.exists(FINETUNED_MODEL_PATH):
                 raise FileNotFoundError(f"Fine-tuned model directory not found: {FINETUNED_MODEL_PATH}")

            print(f"[INFO] Loading fine-tuned model and tokenizer from: {FINETUNED_MODEL_PATH}")
            tokenizer = GPT2TokenizerFast.from_pretrained(FINETUNED_MODEL_PATH)
            model = GPT2LMHeadModel.from_pretrained(FINETUNED_MODEL_PATH)

            # Ensure padding token is set
            if tokenizer.pad_token is None:
                 tokenizer.pad_token = tokenizer.eos_token
                 print("[INFO] Set tokenizer pad_token to eos_token.")

            # Move model to GPU if available
            model.to(GPU_DEVICE)
            model.eval() # Set model to evaluation mode
            print(f"[INFO] Fine-tuned model loaded successfully to {GPU_DEVICE}.")

        except FileNotFoundError as e:
             print(f"[ERROR] {e}")
             print("[ERROR] Please ensure the model has been trained (run in TRAIN mode) or the path is correct.")
             sys.exit(1)
        except Exception as e:
            print(f"[ERROR] Failed to load fine-tuned model: {e}")
            sys.exit(1)

        # 3. Initialize Epistemic Modules & Logger
        print("\n[INFERENCE] === Step 3: Initializing Modules ===")
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
            print("[INFO] All modules initialized successfully.")
        except Exception as e:
             print(f"[ERROR] Failed to initialize modules: {e}")
             sys.exit(1)

        # 4. Start Conversation Loop (Example Usage)
        print("\n[INFERENCE] === Step 4: Starting Interaction ===")
        print("Enter text to start conversation (or type 'quit' to exit):")

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
        print("\n--- Running Predefined Tests ---")
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
        print("\n--- Starting Interactive Loop (type 'quit' to exit) ---")
        while True:
            try:
                user_input = input("You: ")
            except EOFError: # Handle pipe closing etc.
                 print("\n[INFO] Input stream closed. Exiting.")
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

        print("\n--- INFERENCE Mode Execution Finished ---")

    else:
        print(f"[ERROR] Unknown execution mode: {EXECUTION_MODE}")
        parser.print_help()
        sys.exit(1)
