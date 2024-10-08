import src.data_preprocs.data_loading as dl
import polars as pl
import pandas as pd
import pytest

pandas_schema = {
    "age": "int16",
    "campaign": "int16",
    "cons.conf.idx": "Float32",
    "cons.price.idx": "Float32",
    "contact": "object",
    "day_of_week": "object",
    "default": "object",
    "duration": "int16",
    "education": "object",
    "emp.var.rate": "Float32",
    "euribor3m": "Float32",
    "housing": "object",
    "job": "object",
    "loan": "object",
    "marital": "object",
    "month": "object",
    "nr.employed": "Float32",
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
    pandas_loader = dl.DataLoader(file_name="bankmark_samp.csv.gz", class_col="y", schema=pandas_schema)
    polars_loader = dl.DataLoader(file_name="bankmark_samp.csv.gz", class_col="y", schema=polars_schema)
    assert polars_loader._validate_schema() == "polars"
    assert pandas_loader._validate_schema() == "pandas"


def test_validate_framework_framework_and_schema():
    pandas_loader = dl.DataLoader(file_name="bankmark_samp.csv.gz", class_col="y", schema=pandas_schema, data_framework="pandas")
    polars_loader = dl.DataLoader(file_name="bankmark_samp.csv.gz", class_col="y", schema=polars_schema, data_framework="polars")
    assert polars_loader._validate_framework() == "polars"
    assert pandas_loader._validate_framework() == "pandas"


def test_validate_framework_schema_only():
    pandas_loader = dl.DataLoader(file_name="bankmark_samp.csv.gz", class_col="y", schema=pandas_schema)
    polars_loader = dl.DataLoader(file_name="bankmark_samp.csv.gz", class_col="y", schema=polars_schema)
    assert pandas_loader._validate_framework() == "pandas"
    assert polars_loader._validate_framework() == "polars"


def test_validate_framework_framework_only():
    pandas_loader = dl.DataLoader(file_name="bankmark_samp.csv.gz", class_col="y", data_framework="pandas")
    polars_loader = dl.DataLoader(file_name="bankmark_samp.csv.gz", class_col="y", data_framework="polars")
    assert pandas_loader._validate_framework() == "pandas"
    assert polars_loader._validate_framework() == "polars"


def test_validate_framework_negative_cases():
    malformed_loader = dl.DataLoader(file_name="bankmark_samp.csv.gz", class_col="y")
    with pytest.raises(ValueError):
        malformed_loader._validate_framework()

    malformed_loader = dl.DataLoader(file_name="bankmark_samp.csv.gz", class_col="y", data_framework=None, schema=None)
    with pytest.raises(ValueError):
        malformed_loader._validate_framework()

    malformed_loader = dl.DataLoader(file_name="bankmark_samp.csv.gz", class_col="y", data_framework="unsupported")
    with pytest.raises(ValueError):
        malformed_loader._validate_framework()


def test_schema_overrides_framework():
    pandas_loader = dl.DataLoader(file_name="bankmark_samp.csv.gz", class_col="y", schema=pandas_schema, data_framework="polars")
    polars_loader = dl.DataLoader(file_name="bankmark_samp.csv.gz", class_col="y", schema=polars_schema, data_framework="pandas")
    malformed_loader = dl.DataLoader(file_name="bankmark_samp.csv.gz", class_col="y", data_framework="unsupported", schema=polars_schema)
    assert pandas_loader._validate_framework() == "pandas"
    assert polars_loader._validate_framework() == "polars"
    assert malformed_loader._validate_framework() == "polars"


def test_load_data_file_to_pandas_no_schema():
    pandas_loader = dl.DataLoader(file_name="bankmark_samp.csv.gz", class_col="y", data_framework="pandas")
    polars_loader = dl.DataLoader(file_name="bankmark_samp.csv.gz", class_col="y", data_framework="polars")
    assert isinstance(pandas_loader._load_data_file_to_pandas(), pd.DataFrame)
    assert isinstance(polars_loader._load_data_file_to_polars(), pl.DataFrame)


def test_load_data_schema_given_polars():
    data_loader = dl.DataLoader(file_name="bankmark_samp.csv.gz", class_col="y", schema=polars_schema)
    data_loader.load()
    assert isinstance(data_loader.container, dl.DataContainer)
    assert isinstance(data_loader.container.features, pl.DataFrame)
    assert isinstance(data_loader.container.target, pl.Series)


def test_load_data_schema_given_pandas():
    data_loader = dl.DataLoader(file_name="bankmark_samp.csv.gz", class_col="y", schema=pandas_schema)
    data_loader.load()
    assert isinstance(data_loader.container, dl.DataContainer)
    assert isinstance(data_loader.container.features, pd.DataFrame)
    assert isinstance(data_loader.container.target, pd.Series)


def test_load_data_framework_given_polars():
    data_loader = dl.DataLoader(file_name="bankmark_samp.csv.gz", class_col="y", data_framework="polars")
    data_loader.load()
    assert isinstance(data_loader.container, dl.DataContainer)
    assert isinstance(data_loader.container.features, pl.DataFrame)
    assert isinstance(data_loader.container.target, pl.Series)


def test_load_data_framework_given_pandas():
    data_loader = dl.DataLoader(file_name="bankmark_samp.csv.gz", class_col="y", data_framework="pandas")
    data_loader.load()
    assert isinstance(data_loader.container, dl.DataContainer)
    assert isinstance(data_loader.container.features, pd.DataFrame)
    assert isinstance(data_loader.container.target, pd.Series)


def test_load_data_no_schema_or_framework():
    with pytest.raises(ValueError):
        data_loader = dl.DataLoader(file_name="bankmark_samp.csv.gz", class_col="y")
        data_loader.load()

    with pytest.raises(ValueError):
        data_loader = dl.DataLoader(file_name="bankmark_samp.csv.gz", class_col="y", schema=None)
        data_loader.load()

    with pytest.raises(ValueError):
        data_loader = dl.DataLoader(file_name="bankmark_samp.csv.gz", class_col="y", data_framework=None)
        data_loader.load()


# schema overrides framework
def test_load_data_both_schema_and_framework():
    data_loader = dl.DataLoader(file_name="bankmark_samp.csv.gz", class_col="y", schema=pandas_schema, data_framework="polars")
    data_loader.load()
    assert isinstance(data_loader.container, dl.DataContainer)
    assert isinstance(data_loader.container.features, pd.DataFrame)
    assert isinstance(data_loader.container.target, pd.Series)

    data_loader = dl.DataLoader(file_name="bankmark_samp.csv.gz", class_col="y", schema=polars_schema, data_framework="pandas")
    data_loader.load()
    assert isinstance(data_loader.container, dl.DataContainer)
    assert isinstance(data_loader.container.features, pl.DataFrame)
    assert isinstance(data_loader.container.target, pl.Series)


def test_create_data_provider_pandas():
    bankmark = dl.create_data_provider(
        name="bankmark",
        file_name="bankmark_samp.csv.gz",
        sample_size=1.0,
        class_col="y",
        positive_class="Yes",
        spiel="",
        data_framework="pandas",
    )
    assert isinstance(bankmark, dl.DataProvider)
    assert isinstance(bankmark.features, pd.DataFrame)
    assert isinstance(bankmark.target, pd.Series)
    assert bankmark.features.shape[0] == bankmark.target.shape[0]


def test_create_data_provider_polars():
    bankmark = dl.create_data_provider(
        name="bankmark",
        file_name="bankmark_samp.csv.gz",
        sample_size=1.0,
        class_col="y",
        positive_class="Yes",
        spiel="",
        data_framework="polars",
    )
    assert isinstance(bankmark, dl.DataProvider)
    assert isinstance(bankmark.features, pl.DataFrame)
    assert isinstance(bankmark.target, pl.Series)
    assert bankmark.features.shape[0] == bankmark.target.shape[0]
