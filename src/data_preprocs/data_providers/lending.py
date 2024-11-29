from src.data_preprocs.data_loading import DataFramework, DataProviderFactory

lending_common_args = {
    "class_col": "loan_status",
    "positive_class": "Fully Paid",
    "spiel": """Data Set Information:
    Originates from: https://www.lendingclub.com/info/download-data.action

    See also:
    https://www.kaggle.com/wordsforthewise/lending-club

    Prepared by Nate George:  https://github.com/nateGeorge/preprocess_lending_club_data
    """,
    "schema": None,
}

samp_common_args = {
    "name": "lending_samp",
    "file_name": "lending_samp.csv.gz",
    "sample_size": 0.1,
}

factory = DataProviderFactory(kwargs=lending_common_args | samp_common_args | {"data_framework": DataFramework.PANDAS})
lending_samp_pd = factory.create_data_provider()

factory = DataProviderFactory(kwargs=lending_common_args | samp_common_args | {"data_framework": DataFramework.POLARS})
lending_samp_pl = factory.create_data_provider()


small_samp_common_args = {
    "name": "lending_small_samp",
    "file_name": "lending_small_samp.csv.gz",
    "sample_size": 0.01,
}

factory = DataProviderFactory(kwargs=lending_common_args | small_samp_common_args | {"data_framework": DataFramework.PANDAS})
lending_small_samp_pd = factory.create_data_provider()

factory = DataProviderFactory(kwargs=lending_common_args | small_samp_common_args | {"data_framework": DataFramework.POLARS})
lending_small_samp_pl = factory.create_data_provider()


tiny_samp_common_args = {
    "name": "lending_tiny_samp",
    "file_name": "lending_tiny_samp.csv.gz",
    "sample_size": 0.0025,
}

factory = DataProviderFactory(kwargs=lending_common_args | tiny_samp_common_args | {"data_framework": DataFramework.PANDAS})
lending_tiny_samp_pd = factory.create_data_provider()

factory = DataProviderFactory(kwargs=lending_common_args | tiny_samp_common_args | {"data_framework": DataFramework.POLARS})
lending_tiny_samp_pl = factory.create_data_provider()
