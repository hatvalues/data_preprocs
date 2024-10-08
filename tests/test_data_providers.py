import polars as pl
import src.data_preprocs.data_providers as dp
from src.data_preprocs.data_loading import DataProvider
from tests.fixture_helper import assert_dict_matches_fixture
from dataclasses import asdict


def get_test_dict(data_provider: DataProvider) -> dict[str, any]:
    shape = data_provider.features.shape
    column_names = data_provider.features.columns.to_list()
    return {k: v for k, v in asdict(data_provider).items() if k not in ("features", "target")} | {"rows": shape[0], "columns": shape[1], "column_names": column_names}


def test_adult_samp():
    assert_dict_matches_fixture(get_test_dict(dp.adult_samp_pd), "adult")


def test_bankmark_samp_pd():
    assert_dict_matches_fixture(get_test_dict(dp.bankmark_samp_pd), "bankmark")


def test_bankmark_samp_pl():
    assert isinstance(dp.bankmark_samp_pl.features, pl.DataFrame)


def test_breast():
    assert_dict_matches_fixture(get_test_dict(dp.breast_pd), "breast")


def test_car():
    assert_dict_matches_fixture(get_test_dict(dp.car_pd), "car")


def test_cardio():
    assert_dict_matches_fixture(get_test_dict(dp.cardio_pd), "cardio")


def test_cervicalh_pd():
    assert_dict_matches_fixture(get_test_dict(dp.cervicalh_pd), "cervicalh")


def test_cervicalh_pl():
    assert isinstance(dp.cervicalh_pl.features, pl.DataFrame)
    assert all(drop_cols not in dp.cervicalh_pl.features for drop_cols in ["Schiller", "Citology", "Biopsy"])


def test_cervicalr_pd():
    assert_dict_matches_fixture(get_test_dict(dp.cervicalr_pd), "cervicalr")


def test_credit():
    assert_dict_matches_fixture(get_test_dict(dp.credit_pd), "credit")


def test_diaretino():
    assert_dict_matches_fixture(get_test_dict(dp.diaretino_pd), "diaretino")


def test_heart():
    assert_dict_matches_fixture(get_test_dict(dp.heart_pd), "heart")

def test_german():
    assert_dict_matches_fixture(get_test_dict(dp.german_pd), "german")

def test_lending_samp():
    assert_dict_matches_fixture(get_test_dict(dp.lending_samp_pd), "lending_samp")

def test_mhtech14_pd():
    assert_dict_matches_fixture(get_test_dict(dp.mhtech14_pd), "mhtech14")

def test_mhtech14_pl():
    assert isinstance(dp.mhtech14_pl.features, pl.DataFrame)
    assert "comments" not in dp.mhtech14_pl.features.columns