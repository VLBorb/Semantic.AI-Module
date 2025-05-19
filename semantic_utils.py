"""Utility helpers extracted from semantic_ai for standalone use."""
from typing import Dict, List, Tuple

SEMANTIC_CODE_LENGTH = 12


def is_valid_semantic_code(code) -> bool:
    return isinstance(code, str) and len(code) == SEMANTIC_CODE_LENGTH and code.isdigit()


def is_valid_pos_succession(pos1: str, pos2: str, succession_matrix: Dict[str, List[str]]) -> bool:
    return pos2 in succession_matrix.get(pos1, [])


def text_to_codes(text: str, word_to_code: Dict[str, str]) -> Tuple[List[str], List[str]]:
    if not isinstance(text, str):
        return [], []
    words = text.lower().split()
    codes = [word_to_code.get(w, "N/A") for w in words]
    unknown_words = [words[i] for i, code in enumerate(codes) if code == "N/A"]
    return codes, unknown_words


def validate_sequence(codes: List[str], succession_matrix: Dict[str, List[str]], prob_matrix: Dict[str, Dict[str, float]],
                      check_polarity: bool = True, check_register: bool = True) -> Tuple[bool, str]:
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
        return False, "; ".join(errors) if errors else "Sequence contains invalid format codes."

    if len(valid_codes) < 2:
        return True, "Sequence valid (too short for pairwise checks)."

    for i in range(len(valid_codes) - 1):
        c1 = valid_codes[i]
        c2 = valid_codes[i+1]
        pos1, pos2 = c1[:2], c2[:2]
        pol1, pol2 = c1[4], c2[4]
        reg1, reg2 = c1[8], c2[8]

        if not is_valid_pos_succession(pos1, pos2, succession_matrix):
            prob = prob_matrix.get(pos1, {}).get(pos2, 0.0)
            if prob < 0.01:
                errors.append(f"POS violation ({i+1}): {pos1} → {pos2} (Prob={prob:.3f})")

        if check_polarity and pol1 != '0' and pol2 != '0' and pol1 != pol2:
            errors.append(f"Polarity violation ({i+1}): {pol1} → {pol2}")

        if check_register and reg1 != '0' and reg2 != '0' and reg1 != reg2:
            errors.append(f"Register violation ({i+1}): {reg1} → {reg2}")

    if errors:
        return False, "; ".join(errors)
    return True, "Sequence valid"
