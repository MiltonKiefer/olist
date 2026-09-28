import pandas as pd
import shap
from scipy import sparse

def prepare_data_for_shap(pipeline, X):

    preprocessor = pipeline.named_steps["preprocessor"]

    X_transformed = preprocessor.transform(X)

    if sparse.issparse(X_transformed):
        X_transformed = X_transformed.toarray()

    # Recuperar nomes das features
    feature_names = preprocessor.get_feature_names_out()

    X_transformed = pd.DataFrame(X_transformed, columns=feature_names, index=X.index)

    return X_transformed

def create_explainer(pipeline):
    model = pipeline.named_steps["model"]
    explainer = shap.Explainer(model)
    return explainer

def calculate_shap_values(explainer, X_transformed):
    shap_values = explainer(X_transformed)
    return shap_values