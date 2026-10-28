import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from loguru import logger
from sklearn.model_selection import StratifiedKFold, cross_val_score

from module_olist.config import METADATA_PATH, MODEL_PATH
from module_olist.modeling.pipeline import (
    create_gradient_boosting_pipeline,
    create_xgboost_pipeline,
    create_lightgbm_pipeline,
)
from module_olist.modeling.split import FEATURES


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


def save_model(
        model,
        model_name: str,
        threshold: float,
        metrics: dict | None = None,
        model_path: Path = MODEL_PATH,
        metadata_path: Path = METADATA_PATH,
) -> tuple[Path, Path]:
    """
    Salva o modelo treinado e os metadados necessários para a inferência.

    O modelo vai em um arquivo joblib. O nome do modelo, o threshold escolhido
    na validação cruzada, as features e as métricas de teste vão em um JSON
    legível, para que a inferência use exatamente a mesma regra de decisão
    que foi avaliada no treino.
    """
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    metadata = {
        "model_name": model_name,
        "threshold": float(threshold),
        "features": FEATURES,
        "test_metrics": {
            name: float(value) for name, value in (metrics or {}).items()
        },
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)

    logger.info(f"Modelo salvo em {model_path}")
    logger.info(f"Metadados salvos em {metadata_path}")
    return model_path, metadata_path

