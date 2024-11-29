from src.data_preprocs.data_loading import DataFramework, DataProviderFactory

readmit_common_args = {
    "class_col": "readmitted",
    "positive_class": "T",
    "spiel": """
    From Kaggle - https://www.kaggle.com/dansbecker/hospital-readmissions
    No further information
    """,
}

readmit_full_args = {"name": "readmit", "sample_size": 1.0, "file_name": "readmit.csv.gz"}
readmit_samp_args = {"name": "readmit_samp", "sample_size": 0.1, "file_name": "readmit_samp.csv.gz"}

factory = DataProviderFactory(kwargs=readmit_common_args | readmit_full_args | {"data_framework": DataFramework.PANDAS})
readmit_pd = factory.create_data_provider()

factory = DataProviderFactory(kwargs=readmit_common_args | readmit_full_args | {"data_framework": DataFramework.POLARS})
readmit_pl = factory.create_data_provider()

factory = DataProviderFactory(kwargs=readmit_common_args | readmit_samp_args | {"data_framework": DataFramework.PANDAS})
readmit_samp_pd = factory.create_data_provider()

factory = DataProviderFactory(kwargs=readmit_common_args | readmit_samp_args | {"data_framework": DataFramework.POLARS})
readmit_samp_pl = factory.create_data_provider()
