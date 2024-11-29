from src.data_preprocs.data_loading import DataFramework, DataProviderFactory


adult_common_args = {
    "class_col": "income",
    "positive_class": ">50K",
    "spiel": """
    Data Description:
    This data was extracted from the adult bureau database found at
    http://www.adult.gov/ftp/pub/DES/www/welcome.html
    Donor: Ronny Kohavi and Barry Becker,
        Data Mining and Visualization
        Silicon Graphics.
        e-mail: ronnyk@sgi.com for questions.
    Split into train-test using MLC++ GenCVFiles (2/3, 1/3 random).
    48842 instances, mix of continuous and discrete    (train=32561, test=16281)
    45222 if instances with unknown values are removed (train=30162, test=15060)
    Duplicate or conflicting instances : 6
    Class probabilities for adult.all file
    Probability for the label '>50K'  : 23.93% / 24.78% (without unknowns)
    Probability for the label '<=50K' : 76.07% / 75.22% (without unknowns)
    Extraction was done by Barry Becker from the 1994 adult database.  A set of
    reasonably clean records was extracted using the following conditions:
    ((AAGE>16) && (AGI>100) && (AFNLWGT>1)&& (HRSWK>0)),
    """,
    "schema": None,
}

full_set = {"name": "adult", "file_name": "adult.csv.gz", "sample_size": 1.0}
samp_set = {"name": "adult_samp", "file_name": "adult_samp.csv.gz", "sample_size": 0.25}
small_samp_set = {"name": "adult_small_samp", "file_name": "adult_small_samp.csv.gz", "sample_size": 0.025}

factory = DataProviderFactory(kwargs=adult_common_args | full_set | {"data_framework": DataFramework.PANDAS})
adult_pd = factory.create_data_provider()

factory = DataProviderFactory(kwargs=adult_common_args | full_set | {"data_framework": DataFramework.POLARS})
adult_pd = factory.create_data_provider()

factory = DataProviderFactory(kwargs=adult_common_args | samp_set | {"data_framework": DataFramework.PANDAS})
adult_samp_pd = factory.create_data_provider()

factory = DataProviderFactory(kwargs=adult_common_args | samp_set | {"data_framework": DataFramework.POLARS})
adult_samp_pl = factory.create_data_provider()

factory = DataProviderFactory(kwargs=adult_common_args | small_samp_set | {"data_framework": DataFramework.PANDAS})
adult_small_samp_pd = factory.create_data_provider()

factory = DataProviderFactory(kwargs=adult_common_args | small_samp_set | {"data_framework": DataFramework.POLARS})
adult_small_samp_pl = factory.create_data_provider()
