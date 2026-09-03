import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score

from module_olist.modeling.pipeline import (
    create_gradient_boosting_pipeline,
    create_xgboost_pipeline,
    create_lightgbm_pipeline,
)


def get_models():
    return {
        "Gradient Boosting": create_gradient_boosting_pipeline(),
        "XGBoost": create_xgboost_pipeline(),
        "LightGBM": create_lightgbm_pipeline(),
    }


def cross_validate_models(
    models: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 5,
):
    """
    Realiza validação cruzada estratificada no conjunto de treino.
    Mantém o conjunto de teste final separado para avaliação final.
    """
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    cv_results = {}

    for name, model in models.items():
        scores = cross_val_score(
            estimator=model,
            X=X_train,
            y=y_train,
            cv=cv_splitter,
            scoring="f1",
        )

        cv_results[name] = {
            "scores": scores.tolist(),
            "mean_f1": float(scores.mean()),
            "std_f1": float(scores.std()),
        }

    return cv_results


def train_models(
        X_train: pd.DataFrame,
        y_train: pd.Series
):
    models = get_models()

    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model

    return trained_models


def train_selected_model(
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
):
    """Treina somente o modelo escolhido pela validação cruzada."""
    models = get_models()

    if model_name not in models:
        raise ValueError(f"Modelo desconhecido: {model_name}")

    model = models[model_name]
    model.fit(X_train, y_train)
    return model

