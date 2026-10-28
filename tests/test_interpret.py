import pandas as pd
import pytest

from module_olist.modeling.interpret import (
    aggregate_shap_by_feature,
    calculate_shap_values,
    create_explainer,
    prepare_data_for_shap,
)
from module_olist.modeling.pipeline import create_lightgbm_pipeline
from module_olist.modeling.split import FEATURES


def _build_dataset(n_rows=200):
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


@pytest.fixture(scope="module")
def shap_setup():
    data = _build_dataset()
    X = data[FEATURES]
    model = create_lightgbm_pipeline().fit(X, data["is_late"])

    X_transformed = prepare_data_for_shap(model, X)
    explainer = create_explainer(model)
    shap_values = calculate_shap_values(explainer, X_transformed)
    return X, X_transformed, shap_values


def test_prepare_data_for_shap_uses_readable_column_names(shap_setup):
    X, X_transformed, _ = shap_setup

    assert len(X_transformed) == len(X)
    assert not any("__" in column for column in X_transformed.columns)
    assert "customer_state_SP" in X_transformed.columns


def test_calculate_shap_values_has_one_value_per_cell(shap_setup):
    _, X_transformed, shap_values = shap_setup

    assert shap_values.values.shape == X_transformed.shape


def test_aggregate_shap_by_feature_returns_original_variables(shap_setup):
    _, _, shap_values = shap_setup

    importance = aggregate_shap_by_feature(shap_values)

    assert set(importance.index) <= set(FEATURES)
    assert (importance >= 0).all()
    assert importance.is_monotonic_decreasing
