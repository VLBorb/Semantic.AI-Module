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
