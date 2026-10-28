import importlib.util
import os

# O Keras 3 precisa de um backend. Usa o TensorFlow quando ele estiver instalado
# e o PyTorch caso contrário (o TensorFlow ainda não tem pacote para o Python 3.14).
# Para escolher outro, defina a variável de ambiente KERAS_BACKEND.
if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = (
        "tensorflow" if importlib.util.find_spec("tensorflow") else "torch"
    )

import keras
import matplotlib

matplotlib.use("Agg")  # Gera o gráfico direto em arquivo, sem abrir janelas

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from module_olist.config import FIGURES_DIR, INTERIM_DATA_DIR, REPORTS_DIR
from module_olist.modeling.pipeline import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    create_lightgbm_pipeline,
)
from module_olist.modeling.split import split_data

# Parâmetros do experimento
INITIAL_SIZE = 500      # pedidos rotulados no início
BATCH_SIZE = 250        # pedidos que o modelo "manda rotular" em cada rodada
N_ROUNDS = 10           # rodadas de active learning
SEEDS = (0, 1, 2, 3, 4) # repetições, porque uma única rodada é ruidosa
STRATEGIES = ("random", "uncertainty")


class KerasMLPClassifier:
    """
    Rede neural do Keras (MLP) para o risco de atraso, com a API fit/predict_proba.

    O pré-processamento (padronização das colunas numéricas e one-hot das
    categóricas) é ajustado só com os pedidos rotulados que chegam ao fit,
    então nada do pool ou do teste vaza para o treino.
    """

    def __init__(self, epochs: int = 20, batch_size: int = 64, seed: int = 0):
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed

    def _build_model(self, n_inputs: int) -> keras.Model:
        model = keras.Sequential(
            [
                keras.layers.Input(shape=(n_inputs,)),
                keras.layers.Dense(64, activation="relu"),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(32, activation="relu"),
                keras.layers.Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(optimizer="adam", loss="binary_crossentropy")
        return model

    def fit(self, X: pd.DataFrame, y):
        keras.utils.set_random_seed(self.seed)

        self.preprocessor_ = ColumnTransformer(
            transformers=[
                ("numeric", StandardScaler(), NUMERIC_FEATURES),
                (
                    "categorical",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    CATEGORICAL_FEATURES,
                ),
            ]
        )
        X_transformed = self.preprocessor_.fit_transform(X).astype("float32")
        y = np.asarray(y)

        # Só 8% dos pedidos atrasam: dá mais peso à classe rara, para o modelo
        # aprender com poucos exemplos e a probabilidade 0.5 virar o ponto de dúvida.
        positives = max(int(y.sum()), 1)
        class_weight = {0: 1.0, 1: (len(y) - positives) / positives}

        self.model_ = self._build_model(X_transformed.shape[1])
        self.model_.fit(
            X_transformed,
            y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            class_weight=class_weight,
            verbose=0,
        )
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_transformed = self.preprocessor_.transform(X).astype("float32")
        probabilities = self.model_.predict(
            X_transformed, batch_size=4096, verbose=0
        ).ravel()
        return np.column_stack([1 - probabilities, probabilities])


def create_keras_model(seed: int = 0) -> KerasMLPClassifier:
    return KerasMLPClassifier(seed=seed)


def create_lightgbm_model(seed: int = 0):
    """Pipeline do projeto (pré-processamento + LightGBM), sem mensagens de treino."""
    model = create_lightgbm_pipeline()
    model.set_params(model__verbose=-1, model__random_state=seed)
    return model


# Modelos comparados: o do Keras (exemplo pedido) e o LightGBM do projeto (scikit-learn)
MODELS = {
    "Keras (MLP)": create_keras_model,
    "LightGBM (scikit-learn)": create_lightgbm_model,
}


def uncertainty_scores(probabilities: np.ndarray) -> np.ndarray:
    """
    Mede a incerteza do modelo pela entropia da previsão.

    A entropia é máxima (1.0) quando a probabilidade é 0.5, isto é, quando o
    modelo "não consegue decidir", e cai para 0 quando ele está confiante.
    """
    p = np.clip(probabilities, 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def select_batch(model, X_pool: pd.DataFrame, batch_size: int, strategy: str, rng) -> np.ndarray:
    """
    Escolhe quais pedidos do pool (sem rótulo) devem ser rotulados.

    Args:
        model: modelo treinado com os pedidos rotulados até agora.
        X_pool: features dos pedidos ainda sem rótulo (os rótulos não são usados aqui).
        batch_size: quantos pedidos escolher.
        strategy: "random" (sorteio, a referência) ou "uncertainty" (os mais incertos).
        rng: gerador de números aleatórios do NumPy.

    Return:
        Posições (dentro de X_pool) dos pedidos escolhidos.
    """
    batch_size = min(batch_size, len(X_pool))

    if strategy == "random":
        return rng.choice(len(X_pool), size=batch_size, replace=False)

    if strategy == "uncertainty":
        probabilities = model.predict_proba(X_pool)[:, 1]
        scores = uncertainty_scores(probabilities)
        # Os batch_size pedidos com maior incerteza
        return np.argsort(scores)[::-1][:batch_size]

    raise ValueError(f"Estratégia desconhecida: {strategy}")


def run_active_learning(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    strategy: str,
    seed: int,
    model_factory=create_lightgbm_model,
    initial_size: int = INITIAL_SIZE,
    batch_size: int = BATCH_SIZE,
    n_rounds: int = N_ROUNDS,
) -> pd.DataFrame:
    """
    Simula o active learning: o modelo só "enxerga" o rótulo dos pedidos que escolheu.

    Todos os pedidos de treino já têm rótulo, mas a simulação os esconde: o
    treino usa só os pedidos rotulados e a escolha usa só as features do pool.
    O rótulo de um pedido escolhido é "revelado" (como se um humano o rotulasse)
    e ele passa para o conjunto rotulado.

    Args:
        model_factory: função que recebe a semente e devolve um modelo novo
            (com fit e predict_proba), como create_keras_model.

    Return:
        DataFrame com uma linha por rodada: número de rótulos e desempenho no teste.
    """
    rng = np.random.default_rng(seed)

    # Conjunto inicial estratificado (garante pedidos atrasados), igual para as
    # duas estratégias na mesma semente, para a comparação ser justa.
    positions = np.arange(len(X_train))
    labeled, pool = train_test_split(
        positions,
        train_size=initial_size,
        stratify=y_train,
        random_state=seed,
    )

    history = []
    for round_number in range(n_rounds + 1):
        # Treina um modelo novo somente com os pedidos rotulados
        model = model_factory(seed)
        model.fit(X_train.iloc[labeled], y_train.iloc[labeled])

        # Avalia no conjunto de teste, que nunca é usado para escolher pedidos
        y_proba = model.predict_proba(X_test)[:, 1]
        history.append(
            {
                "strategy": strategy,
                "seed": seed,
                "round": round_number,
                "n_labeled": len(labeled),
                "late_labeled": int(y_train.iloc[labeled].sum()),
                "pr_auc": average_precision_score(y_test, y_proba),
                "roc_auc": roc_auc_score(y_test, y_proba),
            }
        )

        if round_number == n_rounds:
            break

        # O modelo escolhe, entre os pedidos sem rótulo, quais quer rotular
        chosen = select_batch(model, X_train.iloc[pool], batch_size, strategy, rng)

        # "Revela" o rótulo dos escolhidos e move esses pedidos do pool para os rotulados
        labeled = np.concatenate([labeled, pool[chosen]])
        pool = np.delete(pool, chosen)

    return pd.DataFrame(history)


def labels_to_reach(curve: pd.Series, target: float):
    """Menor número de rótulos em que a curva (índice = rótulos) alcança o alvo, ou None."""
    reached = curve[curve >= target]
    return int(reached.index[0]) if len(reached) else None


def plot_results(results: pd.DataFrame, full: dict) -> None:
    """Desempenho no teste × número de rótulos: uma linha por modelo, uma curva por estratégia."""
    labels = {"random": "Aleatória (referência)", "uncertainty": "Incerteza (active learning)"}
    colors = {"random": "#8C8C8C", "uncertainty": "#2B6094"}
    metrics = (("pr_auc", "PR-AUC"), ("roc_auc", "ROC-AUC"))

    fig, axes = plt.subplots(len(MODELS), 2, figsize=(12, 4.5 * len(MODELS)), sharex=True)
    for row, model_name in enumerate(MODELS):
        for col, (metric, metric_label) in enumerate(metrics):
            ax = axes[row][col]
            for strategy in STRATEGIES:
                data = results[(results["model"] == model_name) & (results["strategy"] == strategy)]
                grouped = data.groupby("n_labeled")[metric]
                mean, std = grouped.mean(), grouped.std()
                ax.plot(mean.index, mean, marker="o", label=labels[strategy], color=colors[strategy])
                ax.fill_between(mean.index, mean - std, mean + std, alpha=0.2, color=colors[strategy])
            ax.axhline(
                full[model_name][metric], linestyle="--", color="black", linewidth=1,
                label="Todos os rótulos",
            )
            ax.set_title(f"{model_name}: {metric_label} no teste")
            ax.set_xlabel("Pedidos rotulados")
            ax.set_ylabel(metric_label)
            ax.grid(alpha=0.3)
    axes[0][0].legend()
    fig.suptitle("Active learning no risco de atraso: média de 5 repetições (faixa = desvio-padrão)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "active_learning_curve.png", dpi=200, bbox_inches="tight")
    plt.close()


def main():
    """
    Compara active learning (incerteza) com a escolha aleatória de pedidos a rotular,
    com um modelo do Keras e com o LightGBM do projeto.

    Usa a mesma divisão treino/teste do treino do modelo. Salva as curvas em
    reports/figures e os números em reports/active_learning_results.csv.
    """

    data_path = INTERIM_DATA_DIR / "olist_dataset.csv"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {data_path}. "
            "Rode `python -m module_olist.main` antes do active learning."
        )

    X_train, X_test, y_train, y_test = split_data(pd.read_csv(data_path))
    logger.info(
        f"Treino: {len(X_train):,} pedidos (rótulos escondidos na simulação). "
        f"Teste: {len(X_test):,} pedidos. Backend do Keras: {keras.backend.backend()}."
    )

    runs, full = [], {}
    for model_name, model_factory in MODELS.items():
        # Referência (como no exemplo do Keras): modelo treinado com TODOS os rótulos do treino
        full_proba = model_factory(0).fit(X_train, y_train).predict_proba(X_test)[:, 1]
        full[model_name] = {
            "pr_auc": average_precision_score(y_test, full_proba),
            "roc_auc": roc_auc_score(y_test, full_proba),
        }
        logger.info(
            f"{model_name} com todos os {len(X_train):,} rótulos: "
            f"PR-AUC {full[model_name]['pr_auc']:.3f}, ROC-AUC {full[model_name]['roc_auc']:.3f}"
        )

        for strategy in STRATEGIES:
            for seed in SEEDS:
                logger.info(f"{model_name} | estratégia {strategy} | semente {seed}...")
                history = run_active_learning(
                    X_train, y_train, X_test, y_test, strategy, seed, model_factory
                )
                runs.append(history.assign(model=model_name))
    results = pd.concat(runs, ignore_index=True)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(REPORTS_DIR / "active_learning_results.csv", index=False)
    plot_results(results, full)

    final_labels = int(results["n_labeled"].max())
    for model_name in MODELS:
        model_results = results[results["model"] == model_name]

        # Resumo: desempenho médio por estratégia
        summary = (
            model_results.groupby(["n_labeled", "strategy"])[["pr_auc", "roc_auc"]]
            .mean()
            .unstack("strategy")
            .round(3)
        )
        logger.info(f"{model_name}: desempenho médio no teste\n{summary}")

        # Quantos rótulos a incerteza precisa para igualar o resultado final da escolha aleatória
        mean_pr_auc = model_results.groupby(["strategy", "n_labeled"])["pr_auc"].mean()
        target = mean_pr_auc["random"].loc[final_labels]
        needed = labels_to_reach(mean_pr_auc["uncertainty"], target)
        if needed is None:
            logger.info(
                f"{model_name}: a incerteza não igualou a PR-AUC final da escolha aleatória "
                f"({target:.3f}) com até {final_labels:,} rótulos."
            )
        else:
            logger.success(
                f"{model_name}: a escolha aleatória chega a PR-AUC {target:.3f} com "
                f"{final_labels:,} rótulos; a incerteza chega com {needed:,} "
                f"({1 - needed / final_labels:.0%} menos rótulos)."
            )

    logger.success(f"Curvas salvas em {FIGURES_DIR / 'active_learning_curve.png'}")


if __name__ == "__main__":
    main()
