from src.data_preprocs.data_loading import DataFramework, DataProviderFactory, DataProvider
import polars as pl
import pandas as pd
from copy import deepcopy


mhtech16_common_args = {
    "file_name": "mhtech16.csv.gz",
    "positive_class": "yes",
    "spiel": """
    From Kaggle. The three columns used for treatment are as follows and have been duplicated with shorter keys to make pre-processing easier:
    mh1 = 'Have you ever sought treatment for a mental health issue from a mental health professional?'
    mh2 = 'Have you been diagnosed with a mental health condition by a medical professional?'
    mh3 = 'Do you currently have a mental health disorder?'

    There is also corruption in the file, with the column 'Why or why not?' being duplicated as 'Why or why not?.1', which is somewhat buggy to remove.

    These issues have been addressed in the pre-processing of the data.
    """,
    "sample_size": 1.0,
    "schema": None,
}

treatment_columns = ["mh1", "mh2", "mh3"]
drop_columns = [
    "Have you ever sought treatment for a mental health issue from a mental health professional?",
    "Have you been diagnosed with a mental health condition by a medical professional?",
    "Do you currently have a mental health disorder?",
    "If you have revealed a mental health issue to a client or business contact - do you believe this has impacted you negatively?",
    "Was your anonymity protected if you chose to take advantage of mental health or substance abuse treatment resources with previous employers?",
    "If you have been diagnosed or treated for a mental health disorder - do you ever reveal this to coworkers or employees?",
    "If you have revealed a mental health issue to a coworker or employee - do you believe this has impacted you negatively?",
    "Do you believe your productivity is ever affected by a mental health issue?",
    "If yes - what percentage of your work time (time performing primary or secondary job functions) is affected by a mental health issue?",
    "If you have a mental health issue - do you feel that it interferes with your work when being treated effectively?",
    "If you have a mental health issue - do you feel that it interferes with your work when NOT being treated effectively?",
    "How willing would you be to share with friends and family that you have a mental illness?",
    "If yes - what condition(s) have you been diagnosed with?",
    "If maybe - what condition(s) do you believe you have?",
    "If so - what condition(s) were you diagnosed with?",
]
corrupted_cols = ["Why or why not?", "Why or why not?.1"]


def preproc_extra(data_container: DataProvider, treatment_columns: list) -> DataProvider:
    treatment_columns = deepcopy(treatment_columns)
    treatment_columns.remove(data_container.class_col)
    data_container.spiel = f"This dataset uses '{data_container.class_col}' as the class column, removing the other two options\n" + data_container.spiel
    replacement_dict = {"'yes'": "yes", "'no'": "no"}
    if isinstance(data_container.features, pd.DataFrame):
        data_container.features.replace(replacement_dict, inplace=True)
    elif isinstance(data_container.features, pl.DataFrame):
        text_columns = [col for col, dtype in zip(data_container.features.columns, data_container.features.dtypes) if dtype == pl.Utf8]
        data_container.features = data_container.features.with_columns([pl.col(col).replace(replacement_dict).alias(col) if col in text_columns else pl.col(col) for col in data_container.features.columns])
    return data_container


mh1_common_args = {
    "name": "mh1tech16",
    "class_col": "mh1",
}
factory = DataProviderFactory(kwargs=mhtech16_common_args | mh1_common_args | {"data_framework": DataFramework.PANDAS})
mh1tech16_pd = factory.create_data_provider(drop_cols=drop_columns + treatment_columns + corrupted_cols)
mh1tech16_pd = preproc_extra(mh1tech16_pd, treatment_columns)

factory = DataProviderFactory(kwargs=mhtech16_common_args | mh1_common_args | {"data_framework": DataFramework.POLARS})
mh1tech16_pl = factory.create_data_provider(drop_cols=drop_columns + treatment_columns + corrupted_cols)
mh1tech16_pl = preproc_extra(mh1tech16_pl, treatment_columns)

mh2_common_args = {
    "name": "mh2tech16",
    "class_col": "mh2",
}
factory = DataProviderFactory(kwargs=mhtech16_common_args | mh2_common_args | {"data_framework": DataFramework.PANDAS})
mh2tech16_pd = factory.create_data_provider(drop_cols=drop_columns + treatment_columns + corrupted_cols)
mh2tech16_pd = preproc_extra(mh2tech16_pd, treatment_columns)

factory = DataProviderFactory(kwargs=mhtech16_common_args | mh2_common_args | {"data_framework": DataFramework.POLARS})
mh2tech16_pl = factory.create_data_provider(drop_cols=drop_columns + treatment_columns + corrupted_cols)
mh2tech16_pl = preproc_extra(mh2tech16_pl, treatment_columns)

mh3_common_args = {
    "name": "mh3tech16",
    "class_col": "mh3",
}
factory = DataProviderFactory(kwargs=mhtech16_common_args | mh3_common_args | {"data_framework": DataFramework.PANDAS})
mh3tech16_pd = factory.create_data_provider(drop_cols=drop_columns + treatment_columns + corrupted_cols)
mh3tech16_pd = preproc_extra(mh3tech16_pd, treatment_columns)

factory = DataProviderFactory(kwargs=mhtech16_common_args | mh3_common_args | {"data_framework": DataFramework.POLARS})
mh3tech16_pl = factory.create_data_provider(drop_cols=drop_columns + treatment_columns + corrupted_cols)
mh3tech16_pl = preproc_extra(mh3tech16_pl, treatment_columns)