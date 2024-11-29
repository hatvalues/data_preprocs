from src.data_preprocs.data_loading import DataFramework, DataProviderFactory


heart_common_args = {
    "name": "heart",
    "file_name": "heart.csv.gz",
    "sample_size": 1.0,
    "schema": None,
    "class_col": "HDisease",
    "positive_class": "Yes",
    "spiel": """Creators:
    1. Hungarian Institute of Cardiology. Budapest: Andras Janosi, M.D.
    2. University Hospital, Zurich, Switzerland: William Steinbrunn, M.D.
    3. University Hospital, Basel, Switzerland: Matthias Pfisterer, M.D.
    4. V.A. Medical Center, Long Beach and Cleveland Clinic Foundation: Robert Detrano, M.D., Ph.D.

    Donor:
    David W. Aha (aha '@' ics.uci.edu) (714) 856-8779


    Data Set Information:
    This database contains 76 attributes, but all published experiments refer to using a subset of 14 of them. In particular, the Cleveland database is the only one that has been used by ML researchers to
    this date. The "goal" field refers to the presence of heart disease in the patient. It is integer valued from 0 (no presence) to 4. Experiments with the Cleveland database have concentrated on simply attempting to distinguish presence (values 1,2,3,4) from absence (value 0).
    The names and social security numbers of the patients were recently removed from the database, replaced with dummy values.
    One file has been "processed", that one containing the Cleveland database. All four unprocessed files also exist in this directory.
    To see Test Costs (donated by Peter Turney), please see the folder "Costs"

    Attribute Information:
    Only 14 attributes used:
    1. (Age)
    2. (Sex)
    3. (ChestPain)
    4. (RestBP)
    5. (Chol)
    6. (Fbs)
    7. (RestECG)
    8. (MaxHR)
    9. (ExAng)
    10. (Oldpeak)
    11. (Slope)
    12. (Ca)
    13. (Thal)
    14. (HDisease) (the predicted attribute)
    """,
}

factory = DataProviderFactory(kwargs=heart_common_args | {"data_framework": DataFramework.PANDAS})
heart_pd = factory.create_data_provider()

factory = DataProviderFactory(kwargs=heart_common_args | {"data_framework": DataFramework.POLARS})
heart_pl = factory.create_data_provider()
