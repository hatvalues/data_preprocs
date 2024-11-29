from data_preprocs.data_loading import DataFramework, DataProviderFactory

credit_common_args = {
    "name": "credit",
    "file_name": "credit.csv.gz",
    "sample_size": 1.0,
    "schema": None,
    "class_col": "A16",
    "positive_class": "+",
    "spiel": """
    Data Set Information:

    This file concerns credit card applications. All attribute names and values have been changed to meaningless symbols to protect confidentiality of the data.

    This dataset is interesting because there is a good mix of attributes -- continuous, nominal with small numbers of values, and nominal with larger numbers of values. There are also a few missing values.

    Attribute Information:

    A1:	b, a.
    A2:	continuous.
    A3:	continuous.
    A4:	u, y, l, t.
    A5:	g, p, gg.
    A6:	c, d, cc, i, j, k, m, r, q, w, x, e, aa, ff.
    A7:	v, h, bb, j, n, z, dd, ff, o.
    A8:	continuous.
    A9:	t, f.
    A10:	t, f.
    A11:	continuous.
    A12:	t, f.
    A13:	g, p, s.
    A14:	continuous.
    A15:	continuous.
    A16: +,- (class attribute)
    """,
}

factory = DataProviderFactory(kwargs=credit_common_args | {"data_framework": DataFramework.PANDAS})
credit_pd = factory.create_data_provider()

factory = DataProviderFactory(kwargs=credit_common_args | {"data_framework": DataFramework.POLARS})
credit_pl = factory.create_data_provider()
