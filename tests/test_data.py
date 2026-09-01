import pandas as pd

from module_olist.modeling.pipeline import create_gradient_boosting_pipeline
from module_olist.modeling.split import split_data
from module_olist.modeling.train import cross_validate_models


FEATURES = [
    "promised_days",
    "purchase_hour",
    "purchase_weekday",
    "purchase_month",
    "item_count",
    "seller_count",
    "total_price",
    "total_freight",
    "customer_state",
]


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


def test_split_data_keeps_expected_sizes_and_stratification():
    dataset = _build_dataset(120)

    X_train, X_test, y_train, y_test = split_data(dataset)

    assert len(X_train) == 96
    assert len(X_test) == 24
    assert len(y_train) == 96
    assert len(y_test) == 24
    assert abs(y_train.mean() - y_test.mean()) < 0.15


def test_cross_validate_models_returns_cv_summary():
    dataset = _build_dataset(120)
    X_train, _, y_train, _ = split_data(dataset)

    result = cross_validate_models(
        {"Gradient Boosting": create_gradient_boosting_pipeline()},
        X_train,
        y_train,
        cv=3,
    )

    assert "Gradient Boosting" in result
    assert "mean_f1" in result["Gradient Boosting"]
    assert "std_f1" in result["Gradient Boosting"]
    assert len(result["Gradient Boosting"]["scores"]) == 3
