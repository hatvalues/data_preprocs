from src.data_preprocs.data_loading import DataContainer, DataLoader, DataProviderFactory, DataProvider, DataFramework
import polars as pl
import pandas as pd
import pytest

pandas_schema = {
    "age": "int16",
    "campaign": "int16",
    "cons.conf.idx": "float32",
    "cons.price.idx": "float32",
    "contact": "object",
    "day_of_week": "object",
    "default": "object",
    "duration": "int16",
    "education": "object",
    "emp.var.rate": "float32",
    "euribor3m": "float32",
    "housing": "object",
    "job": "object",
    "loan": "object",
    "marital": "object",
    "month": "object",
    "nr.employed": "float32",
    "pdays": "int16",
    "poutcome": "object",
    "previous": "int16",
    "y": "object",
}

polars_schema = {
    "age": pl.Int16,
    "job": pl.Utf8,
    "marital": pl.Utf8,
    "education": pl.Utf8,
    "default": pl.Utf8,
    "housing": pl.Utf8,
    "loan": pl.Utf8,
    "contact": pl.Utf8,
    "month": pl.Utf8,
    "day_of_week": pl.Utf8,
    "duration": pl.Int16,
    "campaign": pl.Int16,
    "pdays": pl.Int16,
    "previous": pl.Int16,
    "poutcome": pl.Utf8,
    "emp.var.rate": pl.Float32,
    "cons.price.idx": pl.Float32,
    "cons.conf.idx": pl.Float32,
    "euribor3m": pl.Float32,
    "nr.employed": pl.Float32,
    "y": pl.Utf8,
}


def test_validate_schema():
    pandas_loader = DataLoader(file_name="bankmark_samp.csv.gz", class_col="y", schema=pandas_schema)
    polars_loader = DataLoader(file_name="bankmark_samp.csv.gz", class_col="y", schema=polars_schema)
    assert polars_loader._validate_schema() == DataFramework.POLARS
    assert pandas_loader._validate_schema() == DataFramework.PANDAS


def test_validate_framework_framework_and_schema():
    pandas_loader = DataLoader(file_name="bankmark_samp.csv.gz", class_col="y", schema=pandas_schema, data_framework=DataFramework.PANDAS)
    polars_loader = DataLoader(file_name="bankmark_samp.csv.gz", class_col="y", schema=polars_schema, data_framework=DataFramework.POLARS)
    assert polars_loader.data_framework == DataFramework.POLARS
    assert pandas_loader.data_framework == DataFramework.PANDAS


def test_validate_framework_schema_only():
    pandas_loader = DataLoader(file_name="bankmark_samp.csv.gz", class_col="y", schema=pandas_schema)
    polars_loader = DataLoader(file_name="bankmark_samp.csv.gz", class_col="y", schema=polars_schema)
    assert pandas_loader.data_framework == DataFramework.PANDAS
    assert polars_loader.data_framework == DataFramework.POLARS


def test_validate_framework_framework_only():
    pandas_loader = DataLoader(file_name="bankmark_samp.csv.gz", class_col="y", data_framework=DataFramework.PANDAS)
    polars_loader = DataLoader(file_name="bankmark_samp.csv.gz", class_col="y", data_framework=DataFramework.POLARS)
    assert pandas_loader.data_framework == DataFramework.PANDAS
    assert polars_loader.data_framework == DataFramework.POLARS


def test_validate_framework_schema_wrong_type():
    with pytest.raises(ValueError):
        malformed_loader = DataLoader(file_name="bankmark_samp.csv.gz", class_col="y", schema=[str, pl.Utf8])


def test_validate_framework_schema_mixed_types():
    with pytest.raises(ValueError):
        malformed_loader = DataLoader(file_name="bankmark_samp.csv.gz", class_col="y", schema={"age": "int16", "size": pl.Utf8})


def test_validate_framework_neither_given():
    with pytest.raises(ValueError):
        malformed_loader = DataLoader(file_name="bankmark_samp.csv.gz", class_col="y", data_framework=None, schema=None)


def test_validate_framework_unsupported():
    with pytest.raises(ValueError):
        malformed_loader = DataLoader(file_name="bankmark_samp.csv.gz", class_col="y", data_framework="unsupported")


def test_schema_overrides_framework():
    pandas_loader = DataLoader(file_name="bankmark_samp.csv.gz", class_col="y", schema=pandas_schema, data_framework=DataFramework.POLARS)
    polars_loader = DataLoader(file_name="bankmark_samp.csv.gz", class_col="y", schema=polars_schema, data_framework=DataFramework.PANDAS)
    malformed_loader = DataLoader(file_name="bankmark_samp.csv.gz", class_col="y", data_framework="unsupported", schema=polars_schema)
    assert pandas_loader.data_framework == DataFramework.PANDAS
    assert polars_loader.data_framework == DataFramework.POLARS
    assert malformed_loader.data_framework == DataFramework.POLARS


def test_file_to_pandas_no_schema():
    pandas_loader = DataLoader(file_name="bankmark_samp.csv.gz", class_col="y", data_framework=DataFramework.PANDAS)
    polars_loader = DataLoader(file_name="bankmark_samp.csv.gz", class_col="y", data_framework=DataFramework.POLARS)
    assert isinstance(pandas_loader._file_to_pandas(), pd.DataFrame)
    assert isinstance(polars_loader._file_to_polars(), pl.DataFrame)


def test_schema_given_polars():
    data_loader = DataLoader(file_name="bankmark_samp.csv.gz", class_col="y", schema=polars_schema)

    assert isinstance(data_loader.container, DataContainer)
    assert isinstance(data_loader.container.features, pl.DataFrame)
    assert isinstance(data_loader.container.target, pl.Series)


def test_schema_given_pandas():
    data_loader = DataLoader(file_name="bankmark_samp.csv.gz", class_col="y", schema=pandas_schema)

    assert isinstance(data_loader.container, DataContainer)
    assert isinstance(data_loader.container.features, pd.DataFrame)
    assert isinstance(data_loader.container.target, pd.Series)


def test_framework_given_polars():
    data_loader = DataLoader(file_name="bankmark_samp.csv.gz", class_col="y", data_framework=DataFramework.POLARS)

    assert isinstance(data_loader.container, DataContainer)
    assert isinstance(data_loader.container.features, pl.DataFrame)
    assert isinstance(data_loader.container.target, pl.Series)


def test_framework_given_pandas():
    data_loader = DataLoader(file_name="bankmark_samp.csv.gz", class_col="y", data_framework=DataFramework.PANDAS)

    assert isinstance(data_loader.container, DataContainer)
    assert isinstance(data_loader.container.features, pd.DataFrame)
    assert isinstance(data_loader.container.target, pd.Series)


def test_no_schema_or_framework():
    with pytest.raises(ValueError):
        data_loader = DataLoader(file_name="bankmark_samp.csv.gz", class_col="y")

    with pytest.raises(ValueError):
        data_loader = DataLoader(file_name="bankmark_samp.csv.gz", class_col="y", schema=None)

    with pytest.raises(ValueError):
        data_loader = DataLoader(file_name="bankmark_samp.csv.gz", class_col="y", data_framework=None)


# schema overrides framework
def test_both_schema_and_framework():
    data_loader = DataLoader(file_name="bankmark_samp.csv.gz", class_col="y", schema=pandas_schema, data_framework=DataFramework.POLARS)

    assert isinstance(data_loader.container, DataContainer)
    assert isinstance(data_loader.container.features, pd.DataFrame)
    assert isinstance(data_loader.container.target, pd.Series)

    data_loader = DataLoader(file_name="bankmark_samp.csv.gz", class_col="y", schema=polars_schema, data_framework=DataFramework.PANDAS)

    assert isinstance(data_loader.container, DataContainer)
    assert isinstance(data_loader.container.features, pl.DataFrame)
    assert isinstance(data_loader.container.target, pl.Series)


def test_create_data_provider_pandas():
    factory = DataProviderFactory(
        kwargs=dict(
            name="bankmark",
            file_name="bankmark_samp.csv.gz",
            sample_size=1.0,
            class_col="y",
            positive_class="Yes",
            spiel="",
            data_framework=DataFramework.PANDAS,
        )
    )
    bankmark = factory.create_data_provider()
    assert isinstance(bankmark, DataProvider)
    assert isinstance(bankmark.features, pd.DataFrame)
    assert isinstance(bankmark.target, pd.Series)
    assert bankmark.features.shape[0] == bankmark.target.shape[0]


def test_create_data_provider_polars():
    factory = DataProviderFactory(
        kwargs=dict(
            name="bankmark",
            file_name="bankmark_samp.csv.gz",
            sample_size=1.0,
            class_col="y",
            positive_class="Yes",
            spiel="",
            data_framework=DataFramework.POLARS,
        )
    )
    bankmark = factory.create_data_provider()
    assert isinstance(bankmark, DataProvider)
    assert isinstance(bankmark.features, pl.DataFrame)
    assert isinstance(bankmark.target, pl.Series)
    assert bankmark.features.shape[0] == bankmark.target.shape[0]


def factory_init_validations():
    with pytest.raises(ValueError):
        _ = DataProviderFactory(kwargs=dict())

    with pytest.raises(ValueError):
        _ = DataProviderFactory(kwargs=dict(data_framework=None, schema=None))

    with pytest.raises(ValueError):
        _ = DataProviderFactory(kwargs=dict(data_framework="unsupported"))

    with pytest.raises(ValueError):
        _ = DataProviderFactory(
            kwargs=dict(
                file_name="bankmark_samp.csv.gz",
                sample_size=1.0,
                class_col="y",
                positive_class="Yes",
                spiel="",
                data_framework=DataFramework.PANDAS,
            )
        )

    with pytest.raises(ValueError):
        _ = DataProviderFactory(
            kwargs=dict(
                name="bankmark",
                sample_size=1.0,
                class_col="y",
                positive_class="Yes",
                spiel="",
                data_framework=DataFramework.PANDAS,
            )
        )

    with pytest.raises(ValueError):
        _ = DataProviderFactory(
            kwargs=dict(
                name="bankmark",
                file_name="bankmark_samp.csv.gz",
                sample_size=1.1,
                class_col="y",
                positive_class="Yes",
                spiel="",
                data_framework=DataFramework.PANDAS,
            )
        )
