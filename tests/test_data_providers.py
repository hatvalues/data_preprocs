import polars as pl
from src.data_preprocs.data_loading import DataProvider
from tests.fixture_helper import assert_dict_matches_fixture, load_yaml_fixture_file
from dataclasses import asdict
from pytest_unordered import unordered
from src.data_preprocs.data_providers.adult import adult_samp_pd
from src.data_preprocs.data_providers.bankmark import bankmark_samp_pd, bankmark_samp_pl
from src.data_preprocs.data_providers.breast import breast_pd
from src.data_preprocs.data_providers.car import car_pd
from src.data_preprocs.data_providers.cardio import cardio_pd
from src.data_preprocs.data_providers.cervical import cervicalh_pd, cervicalh_pl, cervicalr_pd
from src.data_preprocs.data_providers.credit import credit_pd
from src.data_preprocs.data_providers.diaretino import diaretino_pd
from src.data_preprocs.data_providers.heart import heart_pd
from src.data_preprocs.data_providers.german import german_pd
from src.data_preprocs.data_providers.lending import lending_samp_pd
from src.data_preprocs.data_providers.mhtech14 import mhtech14_pd, mhtech14_pl
from src.data_preprocs.data_providers.mhtech16 import mh1tech16_pd, mh2tech16_pd, mh3tech16_pd
from src.data_preprocs.data_providers.mush import mush_pd
from src.data_preprocs.data_providers.noshow import noshow_pd, noshow_samp_pd, noshow_small_samp_pd
from src.data_preprocs.data_providers.nursery import nursery_pd, nursery_samp_pd, nursery_samp_pl
from src.data_preprocs.data_providers.rcdv import rcdv_pd, rcdv_samp_pd
from src.data_preprocs.data_providers.readmit import readmit_pd
from src.data_preprocs.data_providers.thyroid import thyroid_pd, thyroid_samp_pd
from src.data_preprocs.data_providers.yps import ypssmk_pd, ypssmk_pl, ypsalc_pd, ypsalc_pl
from src.data_preprocs.data_providers.usoc2 import usoc2_pd, usoc2_pl, usoc2_samp_pd, usoc2_samp_pl


def get_test_columns_descriptors(data_provider: DataProvider) -> dict[str, any]:
    return {k: asdict(v) for k, v in data_provider.column_descriptors.items() if len(v.unique_values) <= 10}


def get_test_dict(data_provider: DataProvider) -> dict[str, any]:
    shape = data_provider.features.shape
    column_names = data_provider.features.columns.to_list()
    target_classes = data_provider.target.unique().tolist()
    return {k: v for k, v in asdict(data_provider).items() if k not in ("features", "target", "column_descriptors")} | {"rows": shape[0], "columns": shape[1], "column_names": column_names, "target_classes": target_classes} | get_test_columns_descriptors(data_provider)


def test_adult_samp():
    print(adult_samp_pd.column_descriptors)
    assert_dict_matches_fixture(get_test_dict(adult_samp_pd), "adult")


def test_bankmark_samp_pd():
    assert_dict_matches_fixture(get_test_dict(bankmark_samp_pd), "bankmark")


def test_bankmark_samp_pl():
    assert isinstance(bankmark_samp_pl.features, pl.DataFrame)


def test_breast():
    assert_dict_matches_fixture(get_test_dict(breast_pd), "breast")


def test_car():
    assert_dict_matches_fixture(get_test_dict(car_pd), "car")


def test_cardio():
    assert_dict_matches_fixture(get_test_dict(cardio_pd), "cardio")


def test_cervicalh_pd():
    assert_dict_matches_fixture(get_test_dict(cervicalh_pd), "cervicalh")


def test_cervicalh_pl():
    assert isinstance(cervicalh_pl.features, pl.DataFrame)
    assert all(drop_cols not in cervicalh_pl.features for drop_cols in ["Schiller", "Citology", "Biopsy"])


def test_cervicalr_pd():
    assert_dict_matches_fixture(get_test_dict(cervicalr_pd), "cervicalr")


def test_credit():
    assert_dict_matches_fixture(get_test_dict(credit_pd), "credit")


def test_diaretino():
    assert_dict_matches_fixture(get_test_dict(diaretino_pd), "diaretino")


def test_heart():
    assert_dict_matches_fixture(get_test_dict(heart_pd), "heart")


def test_german():
    assert_dict_matches_fixture(get_test_dict(german_pd), "german")


def test_lending_samp():
    assert_dict_matches_fixture(get_test_dict(lending_samp_pd), "lending_samp")


def test_mhtech14_pd():
    assert_dict_matches_fixture(get_test_dict(mhtech14_pd), "mhtech14")


def test_mhtech14_pl():
    assert isinstance(mhtech14_pl.features, pl.DataFrame)
    assert "comments" not in mhtech14_pl.features.columns


def test_mh1tech16_pd():
    print(get_test_dict(mh1tech16_pd))
    assert_dict_matches_fixture(get_test_dict(mh1tech16_pd), "mh1tech16")


def test_mh2tech16_pd():
    assert_dict_matches_fixture(get_test_dict(mh2tech16_pd), "mh2tech16")


def test_mh3tech16_pd():
    assert_dict_matches_fixture(get_test_dict(mh3tech16_pd), "mh3tech16")


def test_mush():
    assert_dict_matches_fixture(get_test_dict(mush_pd), "mush")


def test_noshow():
    assert_dict_matches_fixture(get_test_dict(noshow_pd), "noshow")


def test_noshowsamp():
    assert_dict_matches_fixture(get_test_dict(noshow_samp_pd), "noshow_samp")


def test_noshow_small_samp():
    assert_dict_matches_fixture(get_test_dict(noshow_small_samp_pd), "noshow_small_samp")


def test_nursery():
    assert_dict_matches_fixture(get_test_dict(nursery_pd), "nursery")


def test_nursery_samp():
    assert_dict_matches_fixture(get_test_dict(nursery_samp_pd), "nursery_samp")


def test_nursury_samp_pl():
    assert isinstance(nursery_samp_pl.features, pl.DataFrame)
    assert nursery_samp_pl.features["children"].unique().to_list() == unordered(["1", "2", "3", "more"])


def test_rcdv():
    assert_dict_matches_fixture(get_test_dict(rcdv_pd), "rcdv")


def test_rcdv_samp():
    assert_dict_matches_fixture(get_test_dict(rcdv_samp_pd), "rcdv_samp")


def test_readmit():
    assert_dict_matches_fixture(get_test_dict(readmit_pd), "readmit")


def test_thyroid():
    assert_dict_matches_fixture(get_test_dict(thyroid_pd), "thyroid")


def test_thyroid_samp():
    assert_dict_matches_fixture(get_test_dict(thyroid_samp_pd), "thyroid_samp")


# handling nan in a list
def quick_parse(value):
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        return [str(val) for val in value]
    return value


def test_ypssmk():
    test_dict = get_test_dict(ypssmk_pd)
    fixture = load_yaml_fixture_file("ypssmk")
    for key in fixture:
        assert quick_parse(test_dict[key]) == quick_parse(fixture[key])


def test_ypssmk_pl():
    assert isinstance(ypssmk_pl.features, pl.DataFrame)
    assert ypssmk_pl.target.unique().to_list() == unordered(["tried", None, "current", "never", "former"])


def test_ypsalc():
    test_dict = get_test_dict(ypsalc_pd)
    fixture = load_yaml_fixture_file("ypsalc")
    for key in fixture:
        assert quick_parse(test_dict[key]) == quick_parse(fixture[key])


def test_ypsalc_pl():
    assert isinstance(ypsalc_pl.features, pl.DataFrame)
    assert ypsalc_pl.target.unique().to_list() == unordered([None, "never", "social", "a lot"])


def test_usoc2():
    assert_dict_matches_fixture(get_test_dict(usoc2_pd), "usoc2")


def test_usoc2_pl():
    assert isinstance(usoc2_pl.features, pl.DataFrame)
    assert usoc2_pl.target.unique().to_list() == unordered(["neutral", "happy", "unhappy"])


def test_usoc2_samp():
    assert_dict_matches_fixture(get_test_dict(usoc2_samp_pd), "usoc2_samp")


def test_usoc2_samp_pl():
    assert isinstance(usoc2_samp_pl.features, pl.DataFrame)
    assert usoc2_samp_pl.target.unique().to_list() == unordered(["neutral", "happy", "unhappy"])
