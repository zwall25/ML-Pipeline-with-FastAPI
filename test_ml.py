import pytest
# TODO: add necessary import
import numpy as np
import pandas as pd
from ml.model import train_model, inference, compute_model_metrics, save_model, load_model

def test_train_model_returns_classifier():
    """
    Test that train_model returns an sklearn classifier after training.
    """
    # Create a tiny fake dataset for testing
    X = pd.DataFrame({
        "age": [25, 45, 35, 29],
        "hours-per-week": [40, 50, 60, 20],
    })
    y = np.array([0, 1, 0, 1])

    model = train_model(X, y)

    # Check: model is not None and has 'predict' method like sklearn estimators
    assert model is not None
    assert hasattr(model, "predict")


def test_inference_output_shape():
    """
    Test that inference returns a prediction array of the expected shape.
    """
    # Fake data
    X = pd.DataFrame({
        "age": [30, 50],
        "hours-per-week": [40, 60],
    })
    y = np.array([0, 1])

    model = train_model(X, y)

    preds = inference(model, X)

    # Check: output shape matches number of rows
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == X.shape[0]


def test_compute_model_metrics_types():
    """
    Test that compute_model_metrics returns precision, recall, and f1 as floats.
    """
    # Simple true/pred arrays
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])

    precision, recall, f1 = compute_model_metrics(y_true, y_pred)

    # Check: outputs are floats
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(f1, float)
