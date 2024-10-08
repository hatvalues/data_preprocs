from dataclasses import dataclass
import polars as pl
import pandas as pd
from enum import Enum
from importlib_resources import files, as_file
from data_preprocs import data_sources
from typing import Optional, Mapping, Union


class DataFramework(Enum):
    POLARS = "polars"
    PANDAS = "pandas"


# DataTypeClass is deprecated in polars 0.11.0 but mypy keeps insisting on it in the type annotations
PolarSchema = Optional[Mapping[str, Union[pl.DataTypeClass, pl.DataType]]]
PandasSchema = Optional[Mapping[str, str]]


def is_polar_schema(arg: dict) -> bool:
    if not isinstance(arg, dict):
        return False
    return all(isinstance(k, str) and isinstance(v, (pl.DataTypeClass, pl.DataType)) for k, v in arg.items())


def is_pandas_schema(arg: dict) -> bool:
    if not isinstance(arg, dict):
        return False
    return all(isinstance(k, str) and isinstance(v, str) for k, v in arg.items())


def validate_schema(arg) -> Optional[str]:
    if not arg:
        return None
    elif is_polar_schema(arg):
        return "polars"
    elif is_pandas_schema(arg):
        return "pandas"
    return None


def validate_framework(data_framework: Optional[str] = None, schema: Optional[Union[PandasSchema, PolarSchema]] = None) -> str:
    validation = validate_schema(schema) or data_framework
    if validation not in DataFramework._value2member_map_:
        raise ValueError("Unsupported data framework")
    if not validation:
        raise ValueError("Either data_framework or schema must be provided")
    return validation


def load_data_file_to_pandas(filename: str, schema: Optional[PandasSchema] = None) -> pd.DataFrame:
    source = files(data_sources).joinpath(filename)
    with as_file(source) as data_file:
        return pd.read_csv(data_file, dtype=schema)


def load_data_file_to_polars(filename: str, schema: Optional[PolarSchema] = None) -> pl.DataFrame:
    source = files(data_sources).joinpath(filename)
    with as_file(source) as data_file:
        return pl.read_csv(data_file, schema=schema)


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
        if validate_framework(self.data_framework, self.schema) == "pandas":
            data = load_data_file_to_pandas(self.file_name, self.schema)  # type: ignore
            self.container = DataContainer(target=pd.Series(data[self.class_col]), features=data.drop(columns=self.class_col, axis=1))
        else:
            data = load_data_file_to_polars(self.file_name, self.schema)
            self.container = DataContainer(target=data[self.class_col], features=data.drop(self.class_col))


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



def create_data_provider(
    file_name: str,
    name: str,
    class_col: str,
    positive_class: str,
    spiel: str,
    sample_size: float = 1.0,
    data_framework: Optional[str] = "pandas",
    schema: Optional[Union[PandasSchema, PolarSchema]] = None,
) -> DataProvider:
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
