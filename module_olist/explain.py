import matplotlib
matplotlib.use("Agg")  # Gera os gráficos direto em arquivo, sem abrir janelas

import matplotlib.pyplot as plt
import pandas as pd
import shap
from loguru import logger

from module_olist.config import FIGURES_DIR, PROCESSED_DATA_DIR
from module_olist.modeling.interpret import (
    aggregate_shap_by_feature,
    calculate_shap_values,
    create_explainer,
    prepare_data_for_shap,
)
from module_olist.modeling.predict import load_model
from module_olist.modeling.split import FEATURES

SAMPLE_SIZE = 1000


def save_figure(file_name: str) -> None:
    """Salva a figura atual em reports/figures e a fecha."""
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / file_name, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    """
    Explica as previsões do modelo com SHAP.

    Usa uma amostra do conjunto de teste (pedidos que o modelo não viu) e gera
    em reports/figures: importância global, beeswarm, waterfall do pedido de
    maior risco e importância por variável original.
    """

    logger.info("Iniciando explicabilidade do modelo...")

    test_path = PROCESSED_DATA_DIR / "test_dataset.csv"
    if not test_path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {test_path}. "
            "Rode `python -m module_olist.main` antes da explicabilidade."
        )

    # Carrega o conjunto de teste e seleciona uma amostra das features
    data = pd.read_csv(test_path)
    X_sample = data[FEATURES].sample(n=min(SAMPLE_SIZE, len(data)), random_state=42)

    # Carrega o modelo treinado
    model, model_name, threshold = load_model()
    logger.info(f"Modelo: {model_name}")

    # Aplica o pré-processamento
    X_transformed = prepare_data_for_shap(pipeline=model, X=X_sample)

    # Cria o explicador e calcula os valores SHAP
    explainer = create_explainer(pipeline=model)
    shap_values = calculate_shap_values(explainer=explainer, X_transformed=X_transformed)

    logger.success(f"Valores SHAP calculados para {len(X_sample):,} pedidos.")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Importância global: média do impacto de cada coluna na previsão
    shap.plots.bar(shap_values, max_display=10, show=False)
    save_figure("shap_global_bar.png")

    # 2) Beeswarm: cada ponto é um pedido; mostra a direção do efeito de cada coluna
    shap.plots.beeswarm(shap_values, max_display=10, show=False)
    save_figure("shap_beeswarm.png")

    # 3) Waterfall: explica um pedido individual, o de maior risco previsto
    risk = model.predict_proba(X_sample)[:, 1]
    highest_risk_position = int(risk.argmax())
    shap.plots.waterfall(shap_values[highest_risk_position], max_display=10, show=False)
    save_figure("shap_waterfall.png")

    # 4) Importância por variável original (agrupa as colunas geradas pelo one-hot)
    importance = aggregate_shap_by_feature(shap_values)
    ax = importance.sort_values().plot.barh(figsize=(8, 5), color="#2B6094")
    ax.set_xlabel("Média do |valor SHAP|")
    ax.set_ylabel("Variável")
    ax.set_title("Importância por variável original")
    save_figure("shap_importance_by_feature.png")

    logger.info(f"Importância por variável (média do |SHAP|):\n{importance.round(4)}")
    logger.info(
        f"Pedido explicado no waterfall: probabilidade de atraso "
        f"{risk[highest_risk_position]:.1%} (threshold {threshold:.2f})."
    )
    logger.success(f"Gráficos SHAP gerados com sucesso em {FIGURES_DIR}")


if __name__ == "__main__":
    main()
