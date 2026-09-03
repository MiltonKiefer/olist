from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    cross_val_predict,
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

import numpy as np
import pandas as pd

from loguru import logger

from module_olist.modeling.pipeline import (
    create_gradient_boosting_pipeline,
    create_xgboost_pipeline,
    create_lightgbm_pipeline,
)


def summarize_cv(results):
    """
    Resume os resultados da validação cruzada.

    Para cada métrica, calcula:
    - média dos folds;
    - desvio padrão dos folds.
    """

    metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
    ]

    summary = []

    for metric in metrics:

        values = results[f"test_{metric}"]

        summary.append(
            {
                "metric": metric,
                "mean": values.mean(),
                "std": values.std(),
            }
        )

    return pd.DataFrame(summary)


def find_best_threshold(y_true, y_proba):
    """
    Encontra o threshold que maximiza o F1-score.

    O threshold é testado entre 0.01 e 0.99.
    """

    best_threshold = None

    best_f1 = -1
    best_accuracy = None
    best_precision = None
    best_recall = None

    # Testa diferentes thresholds
    for threshold in np.arange(0.01, 1.00, 0.01,):

        # Converte probabilidades em classes
        y_pred = (y_proba >= threshold).astype(int)

        # Calcula as métricas
        accuracy = accuracy_score(y_true, y_pred,)
        precision = precision_score(y_true, y_pred, zero_division=0,)
        recall = recall_score(y_true, y_pred, zero_division=0,)
        f1 = f1_score(y_true, y_pred, zero_division=0,)

        # Verifica se encontrou um F1 melhor
        if f1 > best_f1:

            best_threshold = threshold
            best_accuracy = accuracy
            best_precision = precision
            best_recall = recall
            best_f1 = f1

    return {
        "threshold": best_threshold,
        "accuracy": best_accuracy,
        "precision": best_precision,
        "recall": best_recall,
        "f1": best_f1,
    }


def cross_validate_models(
    X_train,
    y_train,
):
    """
    Executa validação cruzada dos modelos.

    Para cada modelo:
     - Executa Cross Validation;
     - Calcula métricas dos folds;
     - Gera probabilidades Out-of-Fold;
     - Encontra o melhor threshold pelo F1;
     - Compara os modelos pelo F1 OOF otimizado.

    Retorna:
    - nome do melhor modelo;
    - melhor threshold.
    """

    # -------------------------------------------------
    # Pipelines
    # -------------------------------------------------

    pipelines = {
        "Gradient Boosting": create_gradient_boosting_pipeline(),
        "XGBoost": create_xgboost_pipeline(),
        "LightGBM": create_lightgbm_pipeline(),
    }

    # -------------------------------------------------
    # Estratégia de Cross Validation
    # -------------------------------------------------

    kf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    # -------------------------------------------------
    # Métricas
    # -------------------------------------------------

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
    }

    # Guarda os resultados dos modelos
    cv_results = {}

    # -------------------------------------------------
    # Cross Validation
    # -------------------------------------------------

    for name, pipeline in pipelines.items():

        logger.info(
            f"Executando Cross Validation: {name}"
        )

        # ---------------------------------------------
        # Métricas dos folds
        # ---------------------------------------------

        results = cross_validate(
            estimator=pipeline,
            X=X_train,
            y=y_train,
            cv=kf,
            scoring=scoring,
            return_train_score=True,
        )

        # Resume média e desvio padrão
        summary = summarize_cv(
            results
        )

        print("\n")
        print("=" * 60)
        print(
            f"CROSS VALIDATION - {name}"
        )
        print("=" * 60)

        print(summary)

        # ---------------------------------------------
        # Probabilidades Out-of-Fold
        # ---------------------------------------------

        logger.info(f"Calculando probabilidades Out-of-Fold: {name}")

        y_proba_oof = cross_val_predict(
            estimator=pipeline,
            X=X_train,
            y=y_train,
            cv=kf,
            method="predict_proba",
        )[:, 1]

        # ---------------------------------------------
        # Melhor threshold
        # ---------------------------------------------

        threshold_results = (
            find_best_threshold(
                y_true=y_train,
                y_proba=y_proba_oof,
            )
        )

        # ---------------------------------------------
        # Guarda os resultados do modelo
        # ---------------------------------------------

        cv_results[name] = {
            "results": results,
            "summary": summary,
            "threshold": (
                threshold_results[
                    "threshold"
                ]
            ),
            "accuracy": (
                threshold_results[
                    "accuracy"
                ]
            ),
            "precision": (
                threshold_results[
                    "precision"
                ]
            ),
            "recall": (
                threshold_results[
                    "recall"
                ]
            ),
            "f1_oof": (
                threshold_results[
                    "f1"
                ]
            ),
            "pr_auc": (
                results[
                    "test_pr_auc"
                ].mean()
            ),
        }

        # ---------------------------------------------
        # Resultado com threshold otimizado
        # ---------------------------------------------

        logger.success(f"THRESHOLD OTIMIZADO - {name}")

        logger.info(f"Threshold: {threshold_results['threshold']:.2f}")

        logger.info(f"Accuracy: {threshold_results['accuracy']:.3f}")

        logger.info(f"Precision: {threshold_results['precision']:.3f}")

        logger.info(f"Recall: {threshold_results['recall']:.3f}")

        logger.info(f"F1 OOF: {threshold_results['f1']:.3f}")

    # -------------------------------------------------
    # Escolha do melhor modelo
    # -------------------------------------------------

    best_model_name = max(
        cv_results,
        key=lambda name: (
            cv_results[name]["f1_oof"]
        ),
    )

    # Recupera os resultados do vencedor
    best_results = cv_results[
        best_model_name
    ]

    best_threshold = best_results[
        "threshold"
    ]

    best_accuracy = best_results[
        "accuracy"
    ]

    best_precision = best_results[
        "precision"
    ]

    best_recall = best_results[
        "recall"
    ]

    best_f1 = best_results[
        "f1_oof"
    ]

    best_pr_auc = best_results[
        "pr_auc"
    ]

    # -------------------------------------------------
    # Resultado final da seleção
    # -------------------------------------------------

    logger.success(
        f"Melhor modelo: "
        f"{best_model_name}"
    )

    logger.info(f"F1 OOF otimizado: {best_f1:.3f}")

    logger.success("MODELO SELECIONADO")

    logger.info(f"Modelo: {best_model_name}")

    logger.info(f"Threshold: {best_threshold:.2f}")

    logger.info(f"Accuracy OOF: {best_accuracy:.3f}")

    logger.info(f"Precision OOF: {best_precision:.3f}")

    logger.info(f"Recall OOF: {best_recall:.3f}")

    logger.info(f"F1 OOF: {best_f1:.3f}")

    logger.info(f"PR-AUC médio CV: {best_pr_auc:.3f}")

    logger.success("Cross Validation concluído.")

    # -------------------------------------------------
    # Retorno
    # -------------------------------------------------

    return (
        best_model_name,
        best_threshold,
    )