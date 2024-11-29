from data_preprocs.data_loading import DataFramework, DataProviderFactory

car_common_args = {
    "name": "car",
    "file_name": "car.csv.gz",
    "sample_size": 1.0,
    "schema": None,
    "class_col": "acceptability",
    "positive_class": "acc",
    "spiel": """
    M. Bohanec and V. Rajkovic: Knowledge acquisition and explanation for
    multi-attribute decision making. In 8th Intl Workshop on Expert
    Systems and their Applications, Avignon, France. pages 59-78, 1988.

    Within machine-learning, this dataset was used for the evaluation
    of HINT (Hierarchy INduction Tool), which was proved to be able to
    completely reconstruct the original hierarchical model. This,
    together with a comparison with C4.5, is presented in

    B. Zupan, M. Bohanec, I. Bratko, J. Demsar: Machine learning by
    function decomposition. ICML-97, Nashville, TN. 1997 (to appear)
    """,
}

factory = DataProviderFactory(kwargs=car_common_args | {"data_framework": DataFramework.PANDAS})
car_pd = factory.create_data_provider()

factory = DataProviderFactory(kwargs=car_common_args | {"data_framework": DataFramework.POLARS})
car_pl = factory.create_data_provider()
