# Olist

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Brazilian e-commerce public dataset.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         module_olist and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── module_olist   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes module_olist a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

## Uso

1. Coloque os CSVs do Olist em `data/raw/` (`olist_orders_dataset.csv`,
   `olist_order_items_dataset.csv` e `olist_customers_dataset.csv`).
2. Treine o modelo: `python -m module_olist.main`. O modelo vai para
   `models/best_model.joblib` e os metadados (nome do modelo, threshold,
   features e métricas de teste) para `models/metadata.json`.
3. Rode a inferência: `python -m module_olist.inference`. Ela carrega o modelo,
   prevê o risco de atraso dos pedidos do conjunto de teste e salva as
   predições em `data/processed/test_predictions.csv`.
4. Gere a explicabilidade do modelo com SHAP: `python -m module_olist.explain`.
   Os gráficos (importância global, beeswarm, waterfall de um pedido e
   importância por variável) são salvos em `reports/figures/`.

O modelo é um arquivo joblib e só deve ser carregado com a mesma versão do
scikit-learn registrada em `models/metadata.json`.

--------

