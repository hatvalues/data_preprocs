from data_preprocs.data_loading import DataFramework, DataProviderFactory

noshow_common_args = {
    "name": "noshow",
    "file_name": "noshow.csv.gz",
    "sample_size": 1.0,
    "schema": None,
    "class_col": "no_show",
    "positive_class": "Yes",
    "spiel": """Source:
    No further information
    """,
}

noshow_full_args = {"name": "noshow", "sample_size": 1.0, "file_name": "noshow.csv.gz"}

factory = DataProviderFactory(kwargs=noshow_common_args | noshow_full_args | {"data_framework": DataFramework.PANDAS})
noshow_pd = factory.create_data_provider()

factory = DataProviderFactory(kwargs=noshow_common_args | noshow_full_args | {"data_framework": DataFramework.POLARS})
noshow_pl = factory.create_data_provider()

noshow_samp_args = {"name": "noshow_samp", "sample_size": 0.2, "file_name": "noshow_samp.csv.gz"}

factory = DataProviderFactory(kwargs=noshow_common_args | noshow_samp_args | {"data_framework": DataFramework.PANDAS})
noshow_samp_pd = factory.create_data_provider()

factory = DataProviderFactory(kwargs=noshow_common_args | noshow_samp_args | {"data_framework": DataFramework.POLARS})
noshow_samp_pl = factory.create_data_provider()

noshow_small_samp_args = {"name": "noshow_small_samp", "sample_size": 0.02, "file_name": "noshow_small_samp.csv.gz"}

factory = DataProviderFactory(kwargs=noshow_common_args | noshow_small_samp_args | {"data_framework": DataFramework.PANDAS})
noshow_small_samp_pd = factory.create_data_provider()

factory = DataProviderFactory(kwargs=noshow_common_args | noshow_small_samp_args | {"data_framework": DataFramework.POLARS})
noshow_small_samp_pl = factory.create_data_provider()
