from data_preprocs.data_loading import DataFramework, DataProviderFactory


german_common_args = {
    "name": "german",
    "file_name": "german.csv.gz",
    "sample_size": 1.0,
    "schema": None,
    "class_col": "rating",
    "positive_class": "bad",
    "spiel": """Source:
    Professor Dr. Hans Hofmann
    Institut f"ur Statistik und "Okonometrie
    Universit"at Hamburg
    FB Wirtschaftswissenschaften
    Von-Melle-Park 5
    2000 Hamburg 13

    Data Set Information:
    Two datasets are provided. the original dataset, in the form provided by Prof. Hofmann, contains categorical/symbolic attributes and is in the file "german.data".
    For algorithms that need numerical attributes, Strathclyde University produced the file "german.data-numeric". This file has been edited and several indicator variables added to make it suitable for algorithms which cannot cope with categorical variables. Several attributes that are ordered categorical (such as attribute 17) have been coded as integer. This was the form used by StatLog.

    This dataset requires use of a cost matrix:
    . 1 2
    ------
    1 0 1
    -----
    2 5 0

    (1 = Good, 2 = Bad)
    The rows represent the actual classification and the columns the predicted classification.
    It is worse to class a customer as good when they are bad (5), than it is to class a customer as bad when they are good (1).
    """,
}

factory = DataProviderFactory(kwargs=german_common_args | {"data_framework": DataFramework.PANDAS})
german_pd = factory.create_data_provider()

factory = DataProviderFactory(kwargs=german_common_args | {"data_framework": DataFramework.POLARS})
german_pl = factory.create_data_provider()
