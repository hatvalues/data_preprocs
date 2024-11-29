from data_preprocs.data_loading import DataFramework, DataProviderFactory


thyroid_common_args = {
    "schema": None,
    "class_col": "diagnosis",
    "positive_class": "Abnormal",
    "spiel": """Note: diagnosis has been rendered to a two-class column with possible values normal or abnormal.

    Thyroid disease records supplied by the Garavan Institute and J. Ross
    Quinlan, New South Wales Institute, Syndney, Australia. 1987.

    This directory contains the latest version of an archive of thyroid diagnoses
    obtained from the Garvan Institute, consisting of 9172 records from 1984 to
    early 1987.

    Attribute Name			Possible Values
    --------------			---------------
    age:				continuous.
    sex:				M, F.
    on thyroxine:			f, t.
    query on thyroxine:		f, t.
    on antithyroid medication:	f, t.
    sick:				f, t.
    pregnant:			f, t.
    thyroid surgery:		f, t.
    I131 treatment:			f, t.
    query hypothyroid:		f, t.
    query hyperthyroid:		f, t.
    lithium:			f, t.
    goitre:				f, t.
    tumor:				f, t.
    hypopituitary:			f, t.
    psych:				f, t.
    TSH measured:			f, t.
    TSH:				continuous.
    T3 measured:			f, t.
    T3:				continuous.
    TT4 measured:			f, t.
    TT4:				continuous.
    T4U measured:			f, t.
    T4U:				continuous.
    FTI measured:			f, t.
    FTI:				continuous.
    TBG measured:			f, t.
    TBG:				continuous.
    referral source:		WEST, STMW, SVHC, SVI, SVHD, other.

    The original diagnosis consists of a string of letters indicating diagnosed conditions.

    A diagnosis "-" indicates no condition requiring comment.
    A diagnosis of the
    form "X|Y" is interpreted as "consistent with X, but more likely Y".
    The
    conditions are divided into groups where each group corresponds to a class of
    comments.


    Letter	Diagnosis
    ------	---------
    hyperthyroid conditions:
            A	hyperthyroid
            B	T3 toxic
            C	toxic goitre
            D	secondary toxic
    hypothyroid conditions:
            E	hypothyroid
            F	primary hypothyroid
            G	compensated hypothyroid
            H	secondary hypothyroid
    binding protein:
            I	increased binding protein
            J	decreased binding protein
    general health:
            K	concurrent non-thyroidal illness
    replacement therapy:
            L	consistent with replacement therapy
            M	underreplaced
            N	overreplaced
    antithyroid treatment:
            O	antithyroid drugs
            P	I131 treatment
            Q	surgery
    miscellaneous:
            R	discordant assay results
            S	elevated TBG
            T	elevated thyroid hormones

    In experiments with an earlier version of this archive, decision trees were
    derived for the most frequent classes of comments, namely:
    hyperthyroid conditions (A, B, C, D)
    hypothyroid conditions (E, F, G, H)
    binding protein (I, J)
    general health (K)
    replacement therapy (L, M, N)
    discordant results (R)
    """,
}

thyroid_full_args = {"name": "thyroid", "sample_size": 1.0, "file_name": "thyroid.csv.gz"}
thyroid_samp_args = {"name": "thyroid_samp", "sample_size": 0.1, "file_name": "thyroid_samp.csv.gz"}

factory = DataProviderFactory(kwargs=thyroid_common_args | thyroid_full_args | {"data_framework": DataFramework.PANDAS})
thyroid_pd = factory.create_data_provider()

factory = DataProviderFactory(kwargs=thyroid_common_args | thyroid_full_args | {"data_framework": DataFramework.POLARS})
thyroid_pl = factory.create_data_provider()

factory = DataProviderFactory(kwargs=thyroid_common_args | thyroid_samp_args | {"data_framework": DataFramework.PANDAS})
thyroid_samp_pd = factory.create_data_provider()

factory = DataProviderFactory(kwargs=thyroid_common_args | thyroid_samp_args | {"data_framework": DataFramework.POLARS})
thyroid_samp_pl = factory.create_data_provider()
