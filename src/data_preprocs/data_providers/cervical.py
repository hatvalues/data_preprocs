from src.data_preprocs.data_loading import DataFramework, DataProviderFactory

cervical_common_args = {
    "file_name": "cervical.csv.gz",
    "sample_size": 1.0,
    "schema": None,
    "positive_class": "T",
    "spiel": """Data Set Information:
    The dataset was collected at 'Hospital Universitario de Caracas' in Caracas, Venezuela. The dataset comprises demographic information, habits, and historic medical records of 858 patients. Several patients decided not to answer some of the questions because of privacy concerns (missing values).

    Attribute Information:
    (int) Age
    (int) Number of sexual partners
    (int) First sexual intercourse (age)
    (int) Num of pregnancies
    (bool) Smokes
    (bool) Smokes (years)
    (bool) Smokes (packs/year)
    (bool) Hormonal Contraceptives
    (int) Hormonal Contraceptives (years)
    (bool) IUD
    (int) IUD (years)
    (bool) STDs
    (int) STDs (number)
    (bool) STDs:condylomatosis
    (bool) STDs:cervical condylomatosis
    (bool) STDs:vaginal condylomatosis
    (bool) STDs:vulvo-perineal condylomatosis
    (bool) STDs:syphilis
    (bool) STDs:pelvic inflammatory disease
    (bool) STDs:genital herpes
    (bool) STDs:molluscum contagiosum
    (bool) STDs:AIDS
    (bool) STDs:HIV
    (bool) STDs:Hepatitis B
    (bool) STDs:HPV
    (int) STDs: Number of diagnosis
    (int) STDs: Time since first diagnosis
    (int) STDs: Time since last diagnosis
    (bool) Dx:Cancer
    (bool) Dx:CIN
    (bool) Dx:HPV
    (bool) Dx
    (bool) Hinselmann: target variable
    (bool) Schiller: target variable
    (bool) Cytology: target variable
    (bool) Biopsy: target variable


    Relevant Papers:
    Kelwin Fernandes, Jaime S. Cardoso, and Jessica Fernandes. 'Transfer Learning with Partial Observability Applied to Cervical Cancer Screening.' Iberian Conference on Pattern Recognition and Image Analysis. Springer International Publishing, 2017.

    Citation Request:
    Kelwin Fernandes, Jaime S. Cardoso, and Jessica Fernandes. 'Transfer Learning with Partial Observability Applied to Cervical Cancer Screening.' Iberian Conference on Pattern Recognition and Image Analysis. Springer International Publishing, 2017.
    """,
}

h_common_args = {
    "name": "cervicalh",
    "class_col": "Hinselmann",
}
factory = DataProviderFactory(kwargs=cervical_common_args | h_common_args | {"data_framework": DataFramework.PANDAS})
cervicalh_pd = factory.create_data_provider(drop_cols=["Schiller", "Citology", "Biopsy"])
cervicalh_pd.spiel = f"This dataset uses `{h_common_args["class_col"]}` as the class column, removing the other three options\n" + cervicalh_pd.spiel

factory = DataProviderFactory(kwargs=cervical_common_args | h_common_args | {"data_framework": DataFramework.POLARS})
cervicalh_pl = factory.create_data_provider(drop_cols=["Schiller", "Citology", "Biopsy"])
cervicalh_pl.spiel = f"This dataset uses `{h_common_args["class_col"]}` as the class column, removing the other three options\n" + cervicalh_pl.spiel

s_common_args = {
    "name": "cervicals",
    "class_col": "Schiller",
}
factory = DataProviderFactory(kwargs=cervical_common_args | s_common_args | {"data_framework": DataFramework.PANDAS})
cervicals_pd = factory.create_data_provider(drop_cols=["Hinselmann", "Citology", "Biopsy"])
cervicals_pd.spiel = f"This dataset uses `{s_common_args["class_col"]}` as the class column, removing the other three options\n" + cervicals_pd.spiel

factory = DataProviderFactory(kwargs=cervical_common_args | s_common_args | {"data_framework": DataFramework.POLARS})
cervicals_pl = factory.create_data_provider(drop_cols=["Hinselmann", "Citology", "Biopsy"])
cervicals_pl.spiel = f"This dataset uses `{s_common_args["class_col"]}` as the class column, removing the other three options\n" + cervicals_pl.spiel

c_common_args = {
    "name": "cervicalc",
    "class_col": "Citology",
}
factory = DataProviderFactory(kwargs=cervical_common_args | c_common_args | {"data_framework": DataFramework.PANDAS})
cervicalc_pd = factory.create_data_provider(drop_cols=["Hinselmann", "Schiller", "Biopsy"])
cervicalc_pd.spiel = f"This dataset uses `{c_common_args["class_col"]}` as the class column, removing the other three options\n" + cervicalc_pd.spiel

factory = DataProviderFactory(kwargs=cervical_common_args | c_common_args | {"data_framework": DataFramework.POLARS})
cervicalc_pl = factory.create_data_provider(drop_cols=["Hinselmann", "Schiller", "Biopsy"])
cervicalc_pl.spiel = f"This dataset uses `{c_common_args["class_col"]}` as the class column, removing the other three options\n" + cervicalc_pl.spiel

b_common_args = {
    "name": "cervicalb",
    "class_col": "Biopsy",
}
factory = DataProviderFactory(kwargs=cervical_common_args | b_common_args | {"data_framework": DataFramework.PANDAS})
cervicalb_pd = factory.create_data_provider(drop_cols=["Hinselmann", "Schiller", "Citology"])
cervicalb_pd.spiel = f"This dataset uses `{b_common_args["class_col"]}` as the class column, removing the other three options\n" + cervicalb_pd.spiel

factory = DataProviderFactory(kwargs=cervical_common_args | b_common_args | {"data_framework": DataFramework.POLARS})
cervicalb_pl = factory.create_data_provider(drop_cols=["Hinselmann", "Schiller", "Citology"])
cervicalb_pl.spiel = f"This dataset uses `{b_common_args["class_col"]}` as the class column, removing the other three options\n" + cervicalb_pl.spiel

cervicalr_common_args = {
    "file_name": "cervicalr.csv.gz",
    "name": "cervicalr",
    "class_col": "Biopsy",
    "positive_class": "Cancer",
    "spiel": """The cervical cancer dataset contains indicators and risk factors for predicting whether a woman will get cervical cancer. The features include demographic data (such as age), lifestyle, and medical history. The data can be downloaded from the UCI Machine Learning repository and is described by Fernandes, Cardoso, and Fernandes (2017)15.
    The subset of data features used in the book’s examples are:
        Age in years
        Number of sexual partners
        First sexual intercourse (age in years)
        Number of pregnancies
        Smoking yes or no
        Smoking (in years)
        Hormonal contraceptives yes or no
        Hormonal contraceptives (in years)
        Intrauterine device yes or no (IUD)
        Number of years with an intrauterine device (IUD)
        Has patient ever had a sexually transmitted disease (STD) yes or no
        Number of STD diagnoses
        Time since first STD diagnosis
        Time since last STD diagnosis
        The biopsy results “Healthy” or “Cancer”. Target outcome.
    The biopsy serves as the gold standard for diagnosing cervical cancer.
    Missing values for each column were imputed by the mode (most frequent value).

    Fernandes, Kelwin, Jaime S Cardoso, and Jessica Fernandes. “Transfer learning with partial observability applied to cervical cancer screening.” In Iberian Conference on Pattern Recognition and Image Analysis, 243–50. Springer. (2017).
    """,
    "sample_size": 1.0,
    "schema": None,
}

factory = DataProviderFactory(kwargs=cervicalr_common_args | {"data_framework": DataFramework.PANDAS})
cervicalr_pd = factory.create_data_provider()

factory = DataProviderFactory(kwargs=cervicalr_common_args | {"data_framework": DataFramework.POLARS})
cervicalr_pl = factory.create_data_provider()

