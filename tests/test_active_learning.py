import numpy as np
import pandas as pd
import pytest

from module_olist.modeling.active_learning import (
    KerasMLPClassifier,
    labels_to_reach,
    run_active_learning,
    select_batch,
    uncertainty_scores,
)
from module_olist.modeling.pipeline import create_gradient_boosting_pipeline
from module_olist.modeling.split import FEATURES


def _build_dataset(n_rows=300):
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


class FixedModel:
    """Modelo de mentira que devolve probabilidades pré-definidas."""

    def __init__(self, probabilities):
        self.probabilities = np.asarray(probabilities)

    def predict_proba(self, X):
        return np.column_stack([1 - self.probabilities, self.probabilities])


def test_uncertainty_is_highest_at_one_half_and_lowest_at_the_extremes():
    scores = uncertainty_scores(np.array([0.0, 0.1, 0.5, 0.9, 1.0]))

    assert scores.argmax() == 2
    assert scores[2] == pytest.approx(1.0)
    assert scores[1] == pytest.approx(scores[3])
    assert scores[0] < 1e-6 and scores[4] < 1e-6


def test_select_batch_uncertainty_picks_the_most_uncertain_orders():
    probabilities = [0.02, 0.48, 0.95, 0.55, 0.10, 0.50]
    X_pool = pd.DataFrame({"x": range(len(probabilities))})

    chosen = select_batch(
        FixedModel(probabilities), X_pool, batch_size=3, strategy="uncertainty",
        rng=np.random.default_rng(0),
    )

    assert set(chosen) == {5, 1, 3}  # 0.50, 0.48 e 0.55 são os mais próximos de 0.5


def test_select_batch_random_has_no_repeats_and_follows_the_seed():
    X_pool = pd.DataFrame({"x": range(100)})

    first = select_batch(None, X_pool, 20, "random", np.random.default_rng(7))
    second = select_batch(None, X_pool, 20, "random", np.random.default_rng(7))

    assert len(set(first)) == 20
    assert (first == second).all()


def test_select_batch_is_capped_by_the_pool_size():
    X_pool = pd.DataFrame({"x": range(5)})

    chosen = select_batch(None, X_pool, 50, "random", np.random.default_rng(0))

    assert len(chosen) == 5


def test_select_batch_rejects_unknown_strategy():
    with pytest.raises(ValueError):
        select_batch(None, pd.DataFrame({"x": [1]}), 1, "inexistente", np.random.default_rng(0))


@pytest.mark.parametrize("strategy", ["random", "uncertainty"])
def test_run_active_learning_grows_the_labeled_set_every_round(strategy):
    data = _build_dataset()
    X, y = data[FEATURES], data["is_late"]

    history = run_active_learning(
        X.iloc[:240], y.iloc[:240], X.iloc[240:], y.iloc[240:],
        strategy=strategy, seed=0, initial_size=40, batch_size=20, n_rounds=3,
        model_factory=lambda seed: create_gradient_boosting_pipeline(),
    )

    assert list(history["round"]) == [0, 1, 2, 3]
    assert list(history["n_labeled"]) == [40, 60, 80, 100]
    assert history["pr_auc"].between(0, 1).all()


def test_keras_classifier_returns_probabilities_for_both_classes():
    data = _build_dataset()
    X, y = data[FEATURES], data["is_late"]

    model = KerasMLPClassifier(epochs=3, seed=0).fit(X.iloc[:200], y.iloc[:200])
    probabilities = model.predict_proba(X.iloc[200:])

    assert probabilities.shape == (100, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_active_learning_works_with_the_keras_model():
    data = _build_dataset()
    X, y = data[FEATURES], data["is_late"]

    history = run_active_learning(
        X.iloc[:240], y.iloc[:240], X.iloc[240:], y.iloc[240:],
        strategy="uncertainty", seed=0, initial_size=40, batch_size=20, n_rounds=2,
        model_factory=lambda seed: KerasMLPClassifier(epochs=3, seed=seed),
    )

    assert list(history["n_labeled"]) == [40, 60, 80]


def test_labels_to_reach_returns_the_first_point_that_hits_the_target():
    curve = pd.Series([0.1, 0.3, 0.5], index=[500, 750, 1000])

    assert labels_to_reach(curve, 0.3) == 750
    assert labels_to_reach(curve, 0.9) is None
