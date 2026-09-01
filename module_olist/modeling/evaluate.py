import numpy as np
from loguru import logger
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score)

def evaluate_models(models, X_test, y_test):
    for name, model in models.items():
        y_proba = model.predict_proba(X_test)[:, 1]

        best_threshold = None
        best_f1 = -1
        best_precision = None
        best_recall = None

        for threshold in np.arange(0.05, 0.50, 0.01):
            y_pred = (y_proba >= threshold).astype(int)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            if f1 > best_f1:
                # Se F1 for melhor que o Best F1,
                # Atualiz todos os valores
                best_f1 = f1
                best_threshold = threshold
                best_precision = precision
                best_recall = recall

        roc_auc = roc_auc_score(y_test, y_proba)

        logger.info(f"Avaliação do modelo: {name}")
        logger.info(f"Melhor Threshold: {best_threshold:.2f}")
        logger.info(f"Melhor F1: {best_f1:.2f}")
        logger.info(f"Melhor Precision: {best_precision:.2f}")
        logger.info(f"Melhor Recall: {best_recall:.2f}")
        logger.info(f"AUC-ROC: {roc_auc:.3f}")
