"""Models used by semantic_ai."""

import json
import time
import sqlite3
import random
import logging
from collections import Counter, deque
from typing import Dict, List, Optional, Callable, Tuple
import numpy as np
from nltk.corpus import wordnet as wn
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

DEDUCTION_MAX_HOPS = 3

from semantic_utils import (
    is_valid_semantic_code,
    check_full_pos_sequence,
    is_valid_pos_succession,
    codes_to_text,
    semantic_score,
)

logger = logging.getLogger(__name__)


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
                logger.info(f"[Deduction] Warning: Hop {hop+1} consequence '{consequence_text}' has N/A code. Continuing chain.")
                current_premise = consequence_text # Use the text as the next premise
                # How to get codes for the new text premise? This is tricky.
                # Simplification: Continue chain without updating codes, relying only on text match.
                # Or: Stop if code is N/A to ensure codes remain grounded. Let's stop.
                logger.info(f"[Deduction] Stopping chain at hop {hop+1} due to N/A code.")
                break # Stop chain if code is N/A

        # print(f"[Deduction] Chain finished after {len(self.chain)} hops. No valid code deduction found.")
        return None # No valid deduction found within max_hops

    def deduce(self, premise: str, codes: List[str]) -> Optional[Dict]:
        """Deduce a conclusion using inference chains, with fallback to POS succession."""
        if not premise or not codes or not self.word_to_code or not self.succession_matrix:
             logger.info("[Deduction] Error: Engine not properly initialized or invalid input.")
             return None

        # 1. Try chained deduction
        deduction = self._chain_deductions(premise, codes)
        if deduction: # Already checked for N/A code inside _chain_deductions
            logger.info(f"[Deduction] Success via chain: {self.chain[-1]}")
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
                    logger.info(f"[Deduction] Success via POS fallback: {self.chain[-1]}")
                    return fallback_deduction

        # print(f"[Deduction] Failed for premise: '{premise}'")
        return None

class BeliefModule:
    def __init__(self, decay_rate: float = 0.05, max_beliefs: int = 1000):
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
            logger.info(f"[INFO] Contextual Memory DB initialized at: {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"[ERROR] Failed to initialize SQLite DB at {self.db_path}: {e}")
            self.conn = None
            self.cursor = None

    def save_instance(self, instance_id: str, data: Dict):
        """Save instance to cache and SQLite."""
        if not instance_id or not data:
            logger.error("[ERROR] Attempted to save instance with invalid ID or data.")
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
                 logger.error(f"[ERROR] Could not serialize instance data to JSON for ID {instance_id}. Data: {data}. Error: {e}")
            except sqlite3.Error as e:
                logger.error(f"[ERROR] SQLite error saving instance {instance_id}: {e}")
                # Consider rollback or other error handling
            except Exception as e:
                 logger.error(f"[ERROR] Unexpected error saving instance {instance_id} to DB: {e}")
        else:
             logger.warning("[WARN] SQLite connection not available. Instance saved to memory only.")

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
                 logger.error(f"[ERROR] Could not deserialize instance data from DB for ID {instance_id}. Error: {e}")
            except sqlite3.Error as e:
                logger.error(f"[ERROR] SQLite error loading instance {instance_id}: {e}")
            except Exception as e:
                 logger.error(f"[ERROR] Unexpected error loading instance {instance_id} from DB: {e}")

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
             logger.warning("[WARN] DB not available for retrieving recent instances.")
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
                          logger.error(f"[ERROR] Failed to decode recent instance data from DB: {e}")
         except sqlite3.Error as e:
             logger.error(f"[ERROR] SQLite error retrieving recent instances: {e}")
         except Exception as e:
             logger.error(f"[ERROR] Unexpected error retrieving recent instances: {e}")

         return recent_instances


    def __del__(self):
        """Close the database connection upon object deletion."""
        if self.conn:
            try:
                self.conn.close()
                logger.info("[INFO] Contextual Memory DB connection closed.")
            except sqlite3.Error as e:
                logger.error(f"[ERROR] Error closing SQLite connection: {e}")

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
             logger.info("[Ethics Transparency] Warning: word_to_code map not available.")
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
                logger.error(f"[ERROR] Exception during ethical rule {i+1} execution: {e}")
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
        logger.info(f"[INFO] Traceability Logger initialized. Logs will be saved in: {log_path}")

    def _get_log_file_path(self, instance_id: str) -> str:
        """Constructs the path for the instance's log file."""
        # Use instance_id for filename, sanitize if needed
        safe_instance_id = re.sub(r'[^a-zA-Z0-9_\-]', '_', instance_id)
        return os.path.join(self.log_path_base, f"{safe_instance_id}_log.jsonl")

    def log(self, instance_id: str, entry: Dict):
        """Logs an entry for a given instance ID to a JSON Lines file."""
        if not instance_id or not entry:
             logger.info("[Logger ERROR] Attempted to log with invalid instance ID or empty entry.")
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
            logger.info(f"[Logger ERROR] Failed to serialize log entry to JSON for instance {instance_id}. Entry: {entry_with_timestamp}. Error: {e}")
        except IOError as e:
            logger.info(f"[Logger ERROR] Failed to write to log file {file_path}. Error: {e}")
        except Exception as e:
             logger.info(f"[Logger ERROR] Unexpected error writing log for instance {instance_id}: {e}")

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
            logger.info(f"[BLEU ERROR] Unexpected error: {e}")
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
             logger.info(f"[ROUGE ERROR] Unexpected error: {e}")
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

