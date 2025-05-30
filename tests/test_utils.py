codex/set-up-semantic-ai-system-configuration
import unittest

from semantic_utils import (
    is_valid_semantic_code,
    text_to_codes,
    validate_sequence,
)


class UtilsTestCase(unittest.TestCase):
    def test_is_valid_semantic_code(self):
        self.assertTrue(is_valid_semantic_code('012345678901'))
        self.assertFalse(is_valid_semantic_code('abc'))
        self.assertFalse(is_valid_semantic_code('12345678901'))  # 11 digits


    def test_text_to_codes(self):
        vocab = {'hello': '010000000000', 'world': '020000000000'}
        codes, unknown = text_to_codes('Hello world', vocab)
        self.assertEqual(codes, ['010000000000', '020000000000'])
        self.assertEqual(unknown, [])

        codes, unknown = text_to_codes('hello unknown', vocab)
        self.assertEqual(codes, ['010000000000', 'N/A'])
        self.assertEqual(unknown, ['unknown'])


    def test_validate_sequence_pos_check(self):
        succession = {'01': ['02'], '02': ['01']}
        prob_matrix = {'01': {'02': 1.0}, '02': {'01': 1.0}}

        valid_codes = ['010000000000', '020000000000']
        ok, msg = validate_sequence(valid_codes, succession, prob_matrix)
        self.assertTrue(ok)

        invalid_codes = ['010000000000', '010000000000']
        ok, msg = validate_sequence(invalid_codes, succession, prob_matrix)
        self.assertFalse(ok)
        self.assertIn('POS violation', msg)


if __name__ == '__main__':
    unittest.main()
=======
import semantic_ai as sai

# Helper to create codes with specific POS, polarity, and register

def make_code(pos: str, pol: str = "0", reg: str = "0") -> str:
    code = list("0" * sai.SEMANTIC_CODE_LENGTH)
    code[0] = pos[0]
    code[1] = pos[1]
    code[4] = pol
    code[8] = reg
    return "".join(code)

def test_is_valid_semantic_code_valid():
    assert sai.is_valid_semantic_code("123456789012")


def test_is_valid_semantic_code_invalid_length():
    assert not sai.is_valid_semantic_code("12345")


def test_is_valid_semantic_code_non_digit():
    assert not sai.is_valid_semantic_code("abcd12345678")


def test_text_to_codes_with_unknown_word():
    word_to_code = {"cat": "000000000001", "runs": "000000000002"}
    codes, unknown = sai.text_to_codes("cat runs fast", word_to_code)
    assert codes == ["000000000001", "000000000002", "N/A"]
    assert unknown == ["fast"]


def test_validate_sequence_valid():
    succession_matrix = {"01": ["02"], "02": ["03"]}
    prob_matrix = {"01": {"02": 1.0}, "02": {"03": 1.0}}
    codes = [make_code("01"), make_code("02"), make_code("03")]
    valid, _ = sai.validate_sequence(codes, succession_matrix, prob_matrix)
    assert valid


def test_validate_sequence_invalid_pos():
    succession_matrix = {"01": ["02"], "02": ["03"]}
    prob_matrix = {"01": {"02": 1.0}, "02": {"01": 0.0}}
    codes = [make_code("02"), make_code("01")]
    valid, msg = sai.validate_sequence(codes, succession_matrix, prob_matrix)
    assert not valid
    assert "POS violation" in msg or "Invalid" in msg


def test_validate_sequence_polarity_register():
    succession_matrix = {"01": ["02"]}
    prob_matrix = {"01": {"02": 1.0}}
    codes = [make_code("01", pol="1", reg="1"), make_code("02", pol="2", reg="2")]
    valid, msg = sai.validate_sequence(codes, succession_matrix, prob_matrix)
    assert not valid
    assert "Polarity" in msg or "Register" in msg
main
