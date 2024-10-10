import polars as pl
import src.data_preprocs.data_providers as dp
from src.data_preprocs.data_loading import DataProvider
from tests.fixture_helper import assert_dict_matches_fixture, load_yaml_fixture_file
from dataclasses import asdict
from pytest_unordered import unordered


def get_test_dict(data_provider: DataProvider) -> dict[str, any]:
    shape = data_provider.features.shape
    column_names = data_provider.features.columns.to_list()
    target_classes = data_provider.target.unique().tolist()
    return {k: v for k, v in asdict(data_provider).items() if k not in ("features", "target")} | {"rows": shape[0], "columns": shape[1], "column_names": column_names, "target_classes": target_classes}


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


def test_mh1tech16_pd():
    assert_dict_matches_fixture(get_test_dict(dp.mh1tech16_pd), "mh1tech16")


def test_mh2tech16_pd():
    assert_dict_matches_fixture(get_test_dict(dp.mh2tech16_pd), "mh2tech16")


def test_mh3tech16_pd():
    assert_dict_matches_fixture(get_test_dict(dp.mh3tech16_pd), "mh3tech16")


def test_mush():
    assert_dict_matches_fixture(get_test_dict(dp.mush_pd), "mush")


def test_noshow():
    assert_dict_matches_fixture(get_test_dict(dp.noshow_pd), "noshow")


def test_noshowsamp():
    assert_dict_matches_fixture(get_test_dict(dp.noshow_samp_pd), "noshow_samp")


def test_noshow_small_samp():
    assert_dict_matches_fixture(get_test_dict(dp.noshow_small_samp_pd), "noshow_small_samp")

def test_nursery():
    assert_dict_matches_fixture(get_test_dict(dp.nursery_pd), "nursery")

def test_nursery_samp():
    assert_dict_matches_fixture(get_test_dict(dp.nursery_samp_pd), "nursery_samp")

def test_nursury_samp_pl():
    assert isinstance(dp.nursery_samp_pl.features, pl.DataFrame)
    assert dp.nursery_samp_pl.features["children"].unique().to_list() == unordered(["1", "2", "3", "more"])

def test_rcdv():
    assert_dict_matches_fixture(get_test_dict(dp.rcdv_pd), "rcdv")

def test_rcdv_samp():
    assert_dict_matches_fixture(get_test_dict(dp.rcdv_samp_pd), "rcdv_samp")

def test_readmit():
    assert_dict_matches_fixture(get_test_dict(dp.readmit_pd), "readmit")


def test_thyroid():
    assert_dict_matches_fixture(get_test_dict(dp.thyroid_pd), "thyroid")

def test_thyroid_samp():
    assert_dict_matches_fixture(get_test_dict(dp.thyroid_samp_pd), "thyroid_samp")


# handling nan in a list
def quick_parse(value):
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        return [str(val) for val in value]
    return value


def test_ypssmk():
    test_dict = get_test_dict(dp.ypssmk_pd)
    fixture = load_yaml_fixture_file("ypssmk")
    for key in test_dict:
        assert quick_parse(test_dict[key]) == quick_parse(fixture[key])

def test_ypssmk_pl():
    assert isinstance(dp.ypssmk_pl.features, pl.DataFrame)
    assert dp.ypssmk_pl.target.unique().to_list() == unordered(['tried', None, 'current', 'never', 'former'])

def test_ypsalc():
    test_dict = get_test_dict(dp.ypsalc_pd)
    fixture = load_yaml_fixture_file("ypsalc")
    for key in test_dict:
        assert quick_parse(test_dict[key]) == quick_parse(fixture[key])

def test_ypsalc_pl():
    assert isinstance(dp.ypsalc_pl.features, pl.DataFrame)
    assert dp.ypsalc_pl.target.unique().to_list() == unordered([None, 'never', 'social', 'a lot'])