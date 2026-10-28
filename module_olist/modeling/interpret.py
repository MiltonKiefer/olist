import pandas as pd
import shap  # Biblioteca utilizada para explicar as previsões dos modelos (contribuição de cada feature)
from scipy import sparse  # O OneHotEncoder pode gerar esse tipo de matriz para economizar memória.

from module_olist.modeling.split import FEATURES


def prepare_data_for_shap(pipeline, X):
    """
    Aplica o mesmo pré-processamento
    utilizado durante o treinamento.
    """

    # Recupera o pré-processador
    preprocessor = pipeline.named_steps["preprocessor"]

    # Aplica as transformações
    X_transformed = preprocessor.transform(X)

    # Se for matriz esparsa, converte para matriz normal
    if sparse.issparse(X_transformed):
        X_transformed = X_transformed.toarray()

    # Recupera os nomes das features. O ColumnTransformer acrescenta o nome da
    # etapa ("numeric__", "categorical__"); removemos para os gráficos ficarem legíveis.
    feature_names = [
        name.split("__", 1)[-1] for name in preprocessor.get_feature_names_out()
    ]

    # Converte para DataFrame
    X_transformed = pd.DataFrame(
        X_transformed,
        columns=feature_names,
        index=X.index,
    )

    return X_transformed


def create_explainer(pipeline):
    """
    Cria o explicador SHAP
    para o modelo treinado.
    """

    model = pipeline.named_steps["model"]

    # Explicador especializado em modelos baseados em árvores.
    explainer = shap.TreeExplainer(model)

    return explainer


def calculate_shap_values(explainer, X_transformed):
    """
    Calcula as contribuições SHAP
    das observações.
    """

    # Explica o que o modelo faz para as observações fornecidas.
    shap_values = explainer(X_transformed)

    return shap_values


def aggregate_shap_by_feature(shap_values, features=FEATURES):
    """
    Calcula a importância SHAP de cada variável original.

    O OneHotEncoder transforma uma variável (como customer_state) em várias
    colunas (customer_state_SP, customer_state_RJ...). Para responder "o estado
    do cliente influencia o risco?", somamos as contribuições dessas colunas
    em cada pedido e só então tiramos a média do valor absoluto.

    Return:
        Series com a importância média de cada variável original, em ordem decrescente.
    """

    contributions = pd.DataFrame(
        shap_values.values,
        columns=shap_values.feature_names,
    )

    # Associa cada coluna transformada à variável original
    # (coluna igual ao nome da variável ou começando com "<variável>_").
    original_feature = {
        column: next(
            feature
            for feature in features
            if column == feature or column.startswith(f"{feature}_")
        )
        for column in contributions.columns
    }

    # Soma as colunas de cada variável, pedido a pedido
    by_feature = contributions.T.groupby(original_feature).sum().T

    return by_feature.abs().mean().sort_values(ascending=False)
