from src.data_preprocs.data_loading import DataFramework, DataProviderFactory
import polars as pl


nursery_polars_schema = {
    "parents": pl.Categorical,
    "has_nurs": pl.Categorical,
    "form": pl.Categorical,
    "children": pl.Categorical,
    "housing": pl.Categorical,
    "finance": pl.Categorical,
    "social": pl.Categorical,
    "health": pl.Categorical,
    "decision": pl.Categorical,
}


nursery_common_args = {
    "name": "nursery",
    "schema": None,
    "class_col": "decision",
    "positive_class": None,
    "spiel": """Data Description:
    The target is a multinomial response variable.
    Nursery Database was derived from a hierarchical decision model
    originally developed to rank applications for nursery schools. It
    was used during several years in 1980's when there was excessive
    enrollment to these schools in Ljubljana, Slovenia, and the
    rejected applications frequently needed an objective
    explanation. The final decision depended on three subproblems:
    occupation of parents and child's nursery, family structure and
    financial standing, and social and health picture of the family.
    The model was developed within expert system shell for decision
    making DEX (M. Bohanec, V. Rajkovic: Expert system for decision
    making. Sistemica 1(1), pp. 145-157, 1990.).
    """,
}

full_nursery_args = {"name": "nursery", "sample_size": 1.0, "file_name": "nursery.csv.gz"}
samp_nursery_args = {"name": "nursery_samp", "sample_size": 0.2, "file_name": "nursery_samp.csv.gz"}

factory = DataProviderFactory(kwargs=nursery_common_args | full_nursery_args | {"data_framework": DataFramework.PANDAS, "schema": None})
nursery_pd = factory.create_data_provider()

factory = DataProviderFactory(kwargs=nursery_common_args | full_nursery_args | {"data_framework": DataFramework.POLARS, "schema": nursery_polars_schema})
nursery_pl = factory.create_data_provider()

factory = DataProviderFactory(kwargs=nursery_common_args | samp_nursery_args | {"data_framework": DataFramework.PANDAS, "schema": None})
nursery_samp_pd = factory.create_data_provider()

factory = DataProviderFactory(kwargs=nursery_common_args | samp_nursery_args | {"data_framework": DataFramework.POLARS, "schema": nursery_polars_schema})
nursery_samp_pl = factory.create_data_provider()
