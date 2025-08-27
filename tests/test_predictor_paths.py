from bone_age.predictor import get_predictor
import os

def test_model_res():
    p = get_predictor()
    assert p.model is not None

def test_no_double_instantiation():
    a = get_predictor()
    b = get_predictor()
    assert a is b