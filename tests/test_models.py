import sys, types
sys.modules.setdefault('torch', types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False)))

from semantic_ai import semantic_score
from semantic_models import BeliefModule, DeductiveCognitionEngine


def test_belief_evaluate():
    bm = BeliefModule()
    vocab = {'cat': '010000000000', 'run': '020000000000'}
    bm.set_word_to_code(vocab)
    strength = bm.evaluate_belief('cat', ['020000000000'], 1)
    assert strength >= 0


def test_deduction():
    axioms = {'cats run': 'fast'}
    vocab = {'cats': '010000000000', 'run': '020000000000', 'fast': '030000000000'}
    succession = {'01': ['02'], '02': ['03']}
    engine = DeductiveCognitionEngine(axioms, vocab, succession)
    result = engine.deduce('cats run', ['010000000000', '020000000000'])
    assert result['code'] == '030000000000'


def test_semantic_score():
    succession = {'01': ['02'], '02': ['01']}
    good = semantic_score(['010000000000', '020000000000'], succession)
    bad = semantic_score(['010000000000', '010000000000'], succession)
    assert good > bad
