from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder # Transforma dados Categóricos em Numéricos
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

NUMERIC_FEATURES = [
    "promised_days",
    "item_count",
    "seller_count",
    "total_price",
    "total_freight",
]

CATEGORICAL_FEATURES = [
    "purchase_month",
    "purchase_weekday",
    "purchase_hour",
    "purchase_state",
    "customer_state"
]

def create_preprocessor () -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            # Nas colunas numéricas
            ("numeric", "passthrough", NUMERIC_FEATURES),
            # 1º Nome da operação; 2º Tipo de Transformação; 3º Quais colunas serão aplicadas
            # Nas colunas categóricas
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES)
        ]
    )

def create_gradient_boosting_pipeline() -> Pipeline:
    """
    Cria um Pipeline de Gradient Boosting com pré-processamento.

    Retorna um objeto Pipeline que inclui o pré-processador e o modelo Gradient Boosting Classifier.
    """

    preprocessor = create_preprocessor()

    model = GradientBoostingClassifier(
        n_estimators=100, # Número de Árvores
        learning_rate=0.1, # Taxa de Aprendizado
        max_depth=3,
        random_state=42
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

def create_xgboost_pipeline() -> Pipeline:
    """Cria uma pipeline com pré-processamento e classificador XGBoost."""

    preprocessor = create_preprocessor()

    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
        eval_metric="logloss",
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

def create_lightgbm_pipeline() -> Pipeline:
    """Cria uma pipeline com pré-processamento e classificador LightGBM."""

    preprocessor = create_preprocessor()

    model = LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])