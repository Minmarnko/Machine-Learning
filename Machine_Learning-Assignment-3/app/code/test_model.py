from contextvars import copy_context

import pytest
from linear_regression import LinearRegression, NormalPenalty, LassoPenalty, RidgePenalty, ElasticPenalty, Normal, Lasso, Ridge, ElasticNet
import utils



submit = 1
def test_calculate_y_hardcode_1_plus_2_equal_3():
    output = utils.calculate_y_hardcode(1,2, submit)
    assert output == 3

def test_calculate_y_hardcode_2_plus_2_equal_4():
    output = utils.calculate_y_hardcode(2,2, submit)
    assert output == 4

def test_model_input_and_output_shape():
    engine = 1248
    max_power = 83.1
    mileage = 19.36
    input, output = utils.prediction_class_test(engine, max_power, mileage)
    assert input.shape == (1,34), f"Expecting the shape to be (1,34) but got {input.shape=}"
    assert output.shape == (1,), f"Expecting the shape to be (1,4) but got {output.shape=}"