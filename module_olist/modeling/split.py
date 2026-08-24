import pandas as pd
from sklearn.model_selection import train_test_split


FEATURES = [
    "purchase_hour",
    "purchase_weekday",
    "purchase_month",
    "purchase_days",
    "item_count",
    "seller_count",
    "total_price",
    "total_freight",
    "customer_state",
]

TARGET = "is_late"

def split_data(data: pd.DataFrame):
    """
    Divide o dataset em conjunto de treino e teste.
    """

    X = data[FEATURES]
    y = data[TARGET]

    # Faça a divisão entre treino e teste com 20% de teste
    # Obrigatório a utilização do Stratify

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    return X_train, X_test, y_train, y_test