import json

import pandas as pd
import pytest

from module_olist.modeling.pipeline import create_gradient_boosting_pipeline
from module_olist.modeling.predict import load_model, predict
from module_olist.modeling.split import FEATURES
from module_olist.modeling.train import save_model


def _build_dataset(n_rows=120):
    rows = []
    for i in range(n_rows):
        rows.append(
            {
                "promised_days": 3 + (i % 7),
                "purchase_hour": i % 24,
                "purchase_weekday": i % 7,
                "purchase_month": 1 + (i % 12),
                "item_count": 1 + (i % 3),
                "seller_count": 1 + (i % 2),
                "total_price": 50.0 + (i * 1.5),
                "total_freight": 5.0 + (i % 5),
                "customer_state": "SP" if i % 2 == 0 else "RJ",
                "is_late": 1 if i % 3 == 0 else 0,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def saved_model(tmp_path):
    data = _build_dataset()
    model = create_gradient_boosting_pipeline().fit(data[FEATURES], data["is_late"])

    model_path = tmp_path / "best_model.joblib"
    metadata_path = tmp_path / "metadata.json"
    save_model(
        model,
        "Gradient Boosting",
        0.3,
        {"f1": 0.5},
        model_path=model_path,
        metadata_path=metadata_path,
    )
    return model_path, metadata_path, data


def test_save_model_writes_metadata(saved_model):
    _, metadata_path, _ = saved_model

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert metadata["model_name"] == "Gradient Boosting"
    assert metadata["threshold"] == 0.3
    assert metadata["features"] == FEATURES
    assert metadata["test_metrics"] == {"f1": 0.5}


def test_load_model_returns_model_name_and_threshold(saved_model):
    model_path, metadata_path, _ = saved_model

    model, model_name, threshold = load_model(model_path, metadata_path)

    assert model_name == "Gradient Boosting"
    assert threshold == 0.3
    assert hasattr(model, "predict_proba")


def test_predict_returns_probability_and_class(saved_model):
    model_path, metadata_path, data = saved_model
    model, _, threshold = load_model(model_path, metadata_path)

    result = predict(model=model, X=data[FEATURES], threshold=threshold)

    assert list(result.columns) == ["prob_is_late", "prediction"]
    assert len(result) == len(data)
    assert result["prob_is_late"].between(0, 1).all()
    assert set(result["prediction"].unique()) <= {0, 1}


def test_predict_applies_threshold(saved_model):
    model_path, metadata_path, data = saved_model
    model, _, _ = load_model(model_path, metadata_path)

    strict = predict(model=model, X=data[FEATURES], threshold=1.01)
    lenient = predict(model=model, X=data[FEATURES], threshold=0.0)

    assert (strict["prediction"] == 0).all()
    assert (lenient["prediction"] == 1).all()


def test_predict_rejects_missing_columns(saved_model):
    model_path, metadata_path, data = saved_model
    model, _, threshold = load_model(model_path, metadata_path)

    with pytest.raises(ValueError):
        predict(model, data[FEATURES].drop(columns=["customer_state"]), threshold)


def test_load_model_raises_when_files_are_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "nao_existe.joblib", tmp_path / "nao_existe.json")
