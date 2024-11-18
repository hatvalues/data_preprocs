from dataclasses import dataclass, field
from collections import defaultdict
import polars as pl
import pandas as pd
import numpy as np
from enum import Enum
from importlib_resources import files, as_file
from data_preprocs import data_sources
from typing import Optional, Mapping, Union, Any
from pydantic import BaseModel, field_validator


polars_integer_types = {pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64}
polars_float_types = {pl.Float32, pl.Float64}
polars_categorical_types = {pl.Utf8, pl.Boolean, pl.Categorical}
pandas_categorical_types = {"object", "str", "string", "bool", "boolean"}


class DataFramework(Enum):
    POLARS = "polars"
    PANDAS = "pandas"


# DataTypeClass is deprecated in polars 0.11.0 but mypy keeps insisting on it in the type annotations
PolarSchema = Optional[Mapping[str, Union[pl.DataTypeClass, pl.DataType]]]
PandasSchema = Optional[Mapping[str, str]]


@dataclass
class DataContainer:
    features: Union[pd.DataFrame, pl.DataFrame]
    target: Union[pd.Series, pl.Series]


class DataLoader:
    """Base class for data containers."""

    def __init__(
        self,
        file_name: str,
        class_col: str,
        data_framework: Optional[DataFramework] = None,
        schema: Optional[Union[PandasSchema, PolarSchema]] = None,
    ) -> None:
        self.file_name = file_name
        self.class_col = class_col
        self.data_framework = data_framework
        self.schema = schema
        self._validate_framework()
        self._load()
        self._create_column_descriptors()

    def _validate_framework(self):
        framework = self._validate_schema() or self.data_framework
        if not isinstance(framework, DataFramework):
            raise ValueError("Either framework or schema must be provided.")
        self.data_framework = framework

    def _validate_schema(self) -> Optional[str]:
        if not self.schema:
            return None
        elif not isinstance(self.schema, dict):
            raise ValueError("Schema must be a dictionary")
        elif all(isinstance(k, str) and isinstance(v, (pl.DataTypeClass, pl.DataType)) for k, v in self.schema.items()):
            return DataFramework.POLARS
        elif all(isinstance(k, str) and isinstance(v, str) for k, v in self.schema.items()):
            return DataFramework.PANDAS
        raise ValueError("Invalid schema. Probably a mix of framework types in the schema.")

    def _load(self) -> None:
        if self.data_framework == DataFramework.PANDAS:
            data = self._file_to_pandas()  # type: ignore
            self.container = DataContainer(target=pd.Series(data[self.class_col]), features=data.drop(columns=self.class_col, axis=1))
        else:
            data = self._file_to_polars()
            self.container = DataContainer(target=data[self.class_col], features=data.drop(self.class_col))

    def _file_to_pandas(self) -> pd.DataFrame:
        source = files(data_sources).joinpath(self.file_name)
        with as_file(source) as data_file:
            return pd.read_csv(data_file, dtype=self.schema)

    def _file_to_polars(self) -> pl.DataFrame:
        source = files(data_sources).joinpath(self.file_name)
        with as_file(source) as data_file:
            return pl.read_csv(data_file, schema=self.schema)  # type: ignore # type checking was done in _validate_schema

    @staticmethod
    def _column_descriptor() -> dict[str, Any]:
        return {
            "dtype": None,
            "otype": None,
            "unique_values": [],
            "min": None,
            "max": None,
        }

    def _is_numeric_dtype(self, dtype):
        if self.data_framework == DataFramework.PANDAS:
            return np.issubdtype(dtype, np.floating)
        else:
            return dtype in polars_float_types

    def _is_integer_dtype(self, dtype):
        if self.data_framework == DataFramework.PANDAS:
            return np.issubdtype(dtype, np.integer)
        else:
            return dtype in polars_integer_types

    def _is_categorical_dtype(self, dtype):
        if self.data_framework == DataFramework.PANDAS:
            return dtype.name in pandas_categorical_types
        else:  # Polars
            return dtype in polars_categorical_types

    @staticmethod
    def _get_dtype_name(dtype):
        return str(dtype) if isinstance(dtype, (np.generic, pl.DataType)) else dtype.name

    @staticmethod
    def _get_native_value(value):
        # Converts to native Python types from Numpy or Polars types
        return value.item() if isinstance(value, (np.generic, pl.DataType)) else value

    def _extract_min_max(self, series: Union[pd.Series, pl.Series]) -> tuple[Any, Any]:
        return self._get_native_value(series.min()), self._get_native_value(series.max())

    @staticmethod
    def _extract_unique_values(series: Union[pd.Series, pl.Series]) -> list:
        unique_values = series.unique()
        return unique_values.to_list() if isinstance(unique_values, pl.Series) else unique_values.tolist()

    @staticmethod
    def _no_gaps_integer_series(series: Union[pd.Series, pl.Series]) -> bool:
        n_unique = series.n_unique() if isinstance(series, pl.Series) else series.nunique()
        return n_unique == series.max() - series.min() + 1

    def _update_descriptor(self, descriptor, series, otype, min_max=False):
        descriptor["otype"] = otype
        if otype == "categorical":
            descriptor["unique_values"] = self._extract_unique_values(series)
        elif min_max:
            min_val, max_val = self._extract_min_max(series)
            descriptor.update({"min": min_val, "max": max_val})

    def _create_column_descriptors(self) -> None:
        column_descriptors = defaultdict(self._column_descriptor)

        for col in self.container.features.columns:
            series = self.container.features[col]
            dtype = series.dtype
            column_descriptors[col]["dtype"] = self._get_dtype_name(dtype)

            if self._is_categorical_dtype(dtype):
                self._update_descriptor(column_descriptors[col], series, "categorical")
            elif self._is_numeric_dtype(dtype):
                self._update_descriptor(column_descriptors[col], series, "numeric", min_max=True)
            elif self._is_integer_dtype(dtype):
                otype = "ordinal" if self._no_gaps_integer_series(series) else "count"
                self._update_descriptor(column_descriptors[col], series, otype, min_max=True)
                column_descriptors[col]["unique_values"] = self._extract_unique_values(series)
            else:
                column_descriptors[col]["otype"] = "unknown"

        self.column_descriptors = dict(column_descriptors)


@dataclass
class DataProvider:
    """Base class for data providers."""

    name: str
    file_name: str
    class_col: str
    positive_class: str
    spiel: str
    sample_size: float
    features: Union[pd.DataFrame, pl.DataFrame]
    target: Union[pd.Series, pl.Series]
    column_descriptors: dict[str, Any] = field(default_factory=dict)


class DataProviderFactory(BaseModel):
    kwargs: dict[str, Any]

    @field_validator("kwargs")
    def validate_kwargs(cls, kwargs: dict[str, Any]) -> dict[str, Any]:
        if not kwargs:
            raise ValueError("No keyword arguments provided")
        if kwargs.get("data_framework") is None and kwargs.get("schema") is None:
            raise ValueError("Either data_framework or schema must be provided")
        for k, v in kwargs.items():
            if k == "positive_class" and not (isinstance(v, str) or not v):
                raise ValueError("Positive class must be string or None")
            if k in ("file_name", "name", "class_col", "spiel") and not isinstance(v, str):
                raise ValueError(f"Value for {k} must be a string")
            elif k == "sample_size" and not isinstance(v, float) and not 0.0 <= v <= 1.0:
                raise ValueError("Sample size must be a float between 0 and 1")
            elif k == "data_framework" and not isinstance(v, DataFramework):
                raise ValueError("Unsupported data framework")
            elif k == "data_framework" and (v is None or (isinstance(v, str) and v not in DataFramework._value2member_map_)):
                raise ValueError("Unsupported data framework")
            elif k == "schema" and v is not None and not isinstance(v, dict):
                raise ValueError("Schema must be None or a type mapping")
            elif k not in ("file_name", "name", "class_col", "positive_class", "spiel", "sample_size", "data_framework", "schema"):
                raise ValueError(f"Invalid keyword argument {k}")
        return kwargs

    def create_data_provider(self, drop_cols: list = []) -> DataProvider:
        file_name = self.kwargs["file_name"]
        name = self.kwargs["name"]
        class_col = self.kwargs["class_col"]
        positive_class = self.kwargs["positive_class"]
        spiel = self.kwargs["spiel"]
        sample_size = self.kwargs.get("sample_size", 1.0)
        data_framework = self.kwargs.get("data_framework", DataFramework.PANDAS)
        schema = self.kwargs.get("schema", None)

        data_loader = DataLoader(file_name, class_col, data_framework, schema)
        if data_framework == DataFramework.PANDAS:
            features = data_loader.container.features.drop(columns=drop_cols, axis=1, errors="ignore")
        else:
            features = data_loader.container.features.drop(drop_cols, strict=False)

        return DataProvider(
            name=name,
            file_name=file_name,
            class_col=class_col,
            positive_class=positive_class,
            spiel=spiel,
            sample_size=sample_size,
            features=features,
            target=data_loader.container.target,
            column_descriptors=data_loader.column_descriptors,
        )
