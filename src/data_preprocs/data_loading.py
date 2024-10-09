from dataclasses import dataclass
import polars as pl
import pandas as pd
from enum import Enum
from importlib_resources import files, as_file
from data_preprocs import data_sources
from typing import Optional, Mapping, Union, Any
from pydantic import BaseModel, field_validator


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
        data_framework: Optional[str] = None,
        schema: Optional[Union[PandasSchema, PolarSchema]] = None,
    ) -> None:
        self.file_name = file_name
        self.class_col = class_col
        self.data_framework = data_framework
        self.schema = schema

    def load(self) -> None:
        if self._validate_framework() == "pandas":
            data = self._file_to_pandas()  # type: ignore
            self.container = DataContainer(target=pd.Series(data[self.class_col]), features=data.drop(columns=self.class_col, axis=1))
        else:
            data = self._file_to_polars()
            self.container = DataContainer(target=data[self.class_col], features=data.drop(self.class_col))

    def _validate_schema(self) -> Optional[str]:
        if not self.schema:
            return None
        elif not isinstance(self.schema, dict):
            raise ValueError("Schema must be a dictionary")
        elif all(isinstance(k, str) and isinstance(v, (pl.DataTypeClass, pl.DataType)) for k, v in self.schema.items()):
            return "polars"
        elif all(isinstance(k, str) and isinstance(v, str) for k, v in self.schema.items()):
            return "pandas"
        return None

    def _validate_framework(self) -> str:
        validation = self._validate_schema() or self.data_framework
        if validation not in DataFramework._value2member_map_:
            raise ValueError("Unsupported data framework")
        if not validation:
            raise ValueError("Either data_framework or schema must be provided")
        return validation

    def _file_to_pandas(self) -> pd.DataFrame:
        source = files(data_sources).joinpath(self.file_name)
        with as_file(source) as data_file:
            return pd.read_csv(data_file, dtype=self.schema)

    def _file_to_polars(self) -> pl.DataFrame:
        source = files(data_sources).joinpath(self.file_name)
        with as_file(source) as data_file:
            return pl.read_csv(data_file, schema=self.schema)  # type: ignore # type checking was done in _validate_schema


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


class DataProviderFactory(BaseModel):
    kwargs: dict[str, Any]

    @field_validator("kwargs")
    def validate_kwargs(cls, kwargs: dict[str, Any]) -> dict[str, Any]:
        if not kwargs:
            raise ValueError("No keyword arguments provided")
        if kwargs.get("data_framework") is None and kwargs.get("schema") is None:
            raise ValueError("Either data_framework or schema must be provided")
        for k, v in kwargs.items():
            if k in ("file_name", "name", "class_col", "positive_class", "spiel") and not isinstance(v, str):
                raise ValueError(f"Value for {k} must be a string")
            elif k == "sample_size" and not isinstance(v, float) and not 0.0 <= v <= 1.0:
                raise ValueError("Sample size must be a float between 0 and 1")
            elif k == "data_framework" and v is not None and v not in DataFramework._value2member_map_:
                raise ValueError("Unsupported data framework")
            elif k == "schema" and v is not None and not isinstance(v, dict):
                raise ValueError("Schema must be None or a type mapping")
            elif k not in ("file_name", "name", "class_col", "positive_class", "spiel", "sample_size", "data_framework", "schema"):
                raise ValueError(f"Invalid keyword argument {k}")
        return kwargs

    def create_data_provider(self) -> DataProvider:
        file_name = self.kwargs["file_name"]
        name = self.kwargs["name"]
        class_col = self.kwargs["class_col"]
        positive_class = self.kwargs["positive_class"]
        spiel = self.kwargs["spiel"]
        sample_size = self.kwargs.get("sample_size", 1.0)
        data_framework = self.kwargs.get("data_framework", "pandas")
        schema = self.kwargs.get("schema", None)

        data_loader = DataLoader(file_name, class_col, data_framework, schema)
        data_loader.load()
        return DataProvider(
            name=name,
            file_name=file_name,
            class_col=class_col,
            positive_class=positive_class,
            spiel=spiel,
            sample_size=sample_size,
            features=data_loader.container.features,
            target=data_loader.container.target,
        )
