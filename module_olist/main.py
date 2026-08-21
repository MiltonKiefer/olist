from loguru import logger
from module_olist.config import INTERIM_DATA_DIR, RAW_DATA_DIR
from module_olist.dataset import (
    create_dataset,
    create_target,
    load_data,
    save_data,
)
from module_olist.features import create_feature


def main():
    """
    Executa o pipeline de preparação dos dados.

    O pipeline carrega os arquivos brutos, cria a variável-alvo,
    junta as tabelas, gera as features e salva o resultado
    na pasta data/interim.
    """

    orders_path = RAW_DATA_DIR / "olist_orders_dataset.csv"
    items_path = RAW_DATA_DIR / "olist_order_items_dataset.csv"
    customers_path = RAW_DATA_DIR / "olist_customers_dataset.csv"

    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Carregando os dados brutos...")

    # Carrega os CSVs
    orders, items, customers = load_data(
        orders_path=orders_path,
        items_path=items_path,
        customers_path=customers_path
    )

    # Variável alvo: is_late
    # is_late = 1 se o pedido atrasou, 0 caso contrário.
    logger.info("Criando a variável alvo...")
    orders_with_target = create_target(orders)

    # Agrega os items por orders e junta orders, items e customers em um único dataset.
    dataset = create_dataset(
        orders=orders_with_target,
        items=items,
        customers=customers
    )

    dataset = create_feature(dataset)

    output_path = INTERIM_DATA_DIR / "olist_dataset.csv"
    save_data(dataset, output_path)

    logger.info(f"Pipeline finalizado. Dataset salvo em {output_path}")


if __name__ == "__main__":
    main()