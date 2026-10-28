import pandas as pd
from loguru import logger

from module_olist.config import PROCESSED_DATA_DIR
from module_olist.modeling.predict import load_model, predict
from module_olist.modeling.split import FEATURES, TARGET


def main():
    """
    Executa a inferência com o modelo treinado.

    Usa o conjunto de teste salvo pelo treino (pedidos que o modelo não viu),
    mostra algumas amostras com o valor real ao lado da predição e salva
    as predições de todos os pedidos em data/processed.
    """

    test_path = PROCESSED_DATA_DIR / "test_dataset.csv"
    if not test_path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {test_path}. "
            "Rode `python -m module_olist.main` antes da inferência."
        )

    # Carrega o conjunto de teste (features + valor real)
    data = pd.read_csv(test_path)

    # Carrega o modelo já treinado
    model, model_name, threshold = load_model()

    # Realiza a inferência somente com as features usadas no treinamento
    predictions = predict(
        model=model,
        X=data[FEATURES],
        threshold=threshold,
    )
    results = pd.concat([data, predictions], axis=1)

    # Seleciona algumas amostras para visualização
    sample = results.sample(n=5, random_state=42)

    logger.info(f"Amostras selecionadas para inferência:\n{sample[FEATURES]}\n")
    logger.success(
        "Predições realizadas:\n"
        f"{sample[[TARGET, 'prob_is_late', 'prediction']]}"
    )

    output_path = PROCESSED_DATA_DIR / "test_predictions.csv"
    results.to_csv(output_path, index=False)

    logger.info(
        f"{len(results):,} pedidos avaliados com o modelo {model_name}: "
        f"{results['prediction'].mean():.1%} classificados com risco de atraso "
        f"(threshold {threshold:.2f})."
    )
    logger.success(f"Predições salvas em {output_path}")


if __name__ == "__main__":
    main()
