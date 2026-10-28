import json
from pathlib import Path

import joblib
import pandas as pd
from loguru import logger

from module_olist.config import METADATA_PATH, MODEL_PATH
from module_olist.modeling.split import FEATURES


def load_model(
    model_path: Path = MODEL_PATH,
    metadata_path: Path = METADATA_PATH,
):
    """
    Carrega o modelo treinado e os metadados da inferência.

    Return:
        Tupla com o modelo, o nome do modelo e o threshold de decisão.
    """
    for path in (model_path, metadata_path):
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Arquivo não encontrado: {path}. "
                "Rode `python -m module_olist.main` para treinar e salvar o modelo."
            )

    model = joblib.load(model_path)

    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    threshold = float(metadata["threshold"])
    model_name = metadata["model_name"]

    logger.info(f"Modelo carregado: {model_name}")
    logger.info(f"Threshold carregado: {threshold:.2f}")

    return model, model_name, threshold


def predict(model, X: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Realiza a inferência utilizando o threshold definido na validação.

    Args:
        model: pipeline treinado (pré-processamento + classificador).
        X: DataFrame com as colunas de FEATURES, uma linha por pedido.
        threshold: probabilidade mínima para classificar o pedido como atrasado.

    Return:
        DataFrame com a probabilidade de atraso (prob_is_late) e a classe
        prevista (prediction: 1 = atrasa, 0 = no prazo), com o mesmo índice de X.
    """
    missing = [column for column in FEATURES if column not in X.columns]
    if missing:
        raise ValueError(f"Colunas ausentes para a inferência: {missing}")

    # Probabilidade da classe 1 (pedido atrasado)
    y_proba = model.predict_proba(X[FEATURES])[:, 1]

    # Regra de decisão: atrasa se a probabilidade alcançar o threshold do treino
    y_pred = (y_proba >= float(threshold)).astype(int)

    return pd.DataFrame(
        {
            "prob_is_late": y_proba,
            "prediction": y_pred,
        },
        index=X.index,
    )
