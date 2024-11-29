from data_preprocs.data_loading import DataFramework, DataProviderFactory

cardio_common_args = {
    "name": "cardio",
    "file_name": "cardio.csv.gz",
    "sample_size": 1.0,
    "schema": None,
    "class_col": "NSP",
    "positive_class": "N",
    "spiel": """
    Data Set Information:
    2126 fetal cardiotocograms (CTGs) were automatically processed and the respective diagnostic features measured. The CTGs were also classified by three expert obstetricians and a consensus classification label assigned to each of them. Classification was both with respect to a morphologic pattern (A, B, C. ...) and to a fetal state (N, S, P). Therefore the dataset can be used either for 10-class or 3-class experiments.

    Attribute Information:
    LB - FHR baseline (beats per minute)
    AC - # of accelerations per second
    FM - # of fetal movements per second
    UC - # of uterine contractions per second
    DL - # of light decelerations per second
    DS - # of severe decelerations per second
    DP - # of prolongued decelerations per second
    ASTV - percentage of time with abnormal short term variability
    MSTV - mean value of short term variability
    ALTV - percentage of time with abnormal long term variability
    MLTV - mean value of long term variability
    Width - width of FHR histogram
    Min - minimum of FHR histogram
    Max - Maximum of FHR histogram
    Nmax - # of histogram peaks
    Nzeros - # of histogram zeros
    Mode - histogram mode
    Mean - histogram mean
    Median - histogram median
    Variance - histogram variance
    Tendency - histogram tendency
    CLASS - FHR pattern class code (1 to 10) # alternative class
    NSP - fetal state class code (N=normal; S=suspect; P=pathologic)
    """,
}

factory = DataProviderFactory(kwargs=cardio_common_args | {"data_framework": DataFramework.PANDAS})
cardio_pd = factory.create_data_provider()

factory = DataProviderFactory(kwargs=cardio_common_args | {"data_framework": DataFramework.POLARS})
cardio_pl = factory.create_data_provider()
