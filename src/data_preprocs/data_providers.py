from .data_loading import DataProvider, DataProviderFactory
import polars as pl
import pandas as pd


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

factory = DataProviderFactory(kwargs=adult_common_args | full_set | {"data_framework": "pandas"})
adult_pd = factory.create_data_provider()

factory = DataProviderFactory(kwargs=adult_common_args | full_set | {"data_framework": "polars"})
adult_pd = factory.create_data_provider()

factory = DataProviderFactory(kwargs=adult_common_args | samp_set | {"data_framework": "pandas"})
adult_samp_pd = factory.create_data_provider()

factory = DataProviderFactory(kwargs=adult_common_args | samp_set | {"data_framework": "polars"})
adult_samp_pl = factory.create_data_provider()

factory = DataProviderFactory(kwargs=adult_common_args | small_samp_set | {"data_framework": "pandas"})
adult_small_samp_pd = factory.create_data_provider()

factory = DataProviderFactory(kwargs=adult_common_args | small_samp_set | {"data_framework": "polars"})
adult_small_samp_pl = factory.create_data_provider()


bankmark_common_args = {
    "class_col": "y",
    "positive_class": "Yes",
    "spiel": """
    Data Set Information:
    The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed.

    There are four datasets:
    1) bank-additional-full.csv with all examples (41188) and 20 inputs, ordered by date (from May 2008 to November 2010), very close to the data analyzed in [Moro et al., 2014]
    2) bank-additional.csv with 10% of the examples (4119), randomly selected from 1), and 20 inputs.
    3) bank-full.csv with all examples and 17 inputs, ordered by date (older version of this dataset with less inputs).
    4) bank.csv with 10% of the examples and 17 inputs, randomly selected from 3 (older version of this dataset with less inputs).
    The smallest datasets are provided to test more computationally demanding machine learning algorithms (e.g., SVM).

    The classification goal is to predict if the client will subscribe (yes/no) a term deposit (variable y).


    Attribute Information:

    Input variables:
    # bank client data:
    1 - age (numeric)
    2 - job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
    3 - marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
    4 - education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
    5 - default: has credit in default? (categorical: 'no','yes','unknown')
    6 - housing: has housing loan? (categorical: 'no','yes','unknown')
    7 - loan: has personal loan? (categorical: 'no','yes','unknown')
    # related with the last contact of the current campaign:
    8 - contact: contact communication type (categorical: 'cellular','telephone')
    9 - month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
    10 - day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
    11 - duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
    # other attributes:
    12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
    13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
    14 - previous: number of contacts performed before this campaign and for this client (numeric)
    15 - poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')
    # social and economic context attributes
    16 - emp.var.rate: employment variation rate - quarterly indicator (numeric)
    17 - cons.price.idx: consumer price index - monthly indicator (numeric)
    18 - cons.conf.idx: consumer confidence index - monthly indicator (numeric)
    19 - euribor3m: euribor 3 month rate - daily indicator (numeric)
    20 - nr.employed: number of employees - quarterly indicator (numeric)

    Output variable (desired target):
    21 - y - has the client subscribed a term deposit? (binary: 'yes','no')
    """,
}

bankmark_pandas_schema = {
    "age": "int16",
    "campaign": "int16",
    "cons.conf.idx": "float16",
    "cons.price.idx": "float16",
    "contact": "object",
    "day_of_week": "object",
    "default": "object",
    "duration": "int16",
    "education": "object",
    "emp.var.rate": "float16",
    "euribor3m": "float16",
    "housing": "object",
    "job": "object",
    "loan": "object",
    "marital": "object",
    "month": "object",
    "nr.employed": "float16",
    "pdays": "int16",
    "poutcome": "object",
    "previous": "int16",
    "y": "object",
}


bankmark_polars_schema = {
    "age": pl.Int16,
    "job": pl.Utf8,
    "marital": pl.Utf8,
    "education": pl.Utf8,
    "default": pl.Utf8,
    "housing": pl.Utf8,
    "loan": pl.Utf8,
    "contact": pl.Utf8,
    "month": pl.Utf8,
    "day_of_week": pl.Utf8,
    "duration": pl.Int16,
    "campaign": pl.Int16,
    "pdays": pl.Int16,
    "previous": pl.Int16,
    "poutcome": pl.Utf8,
    "emp.var.rate": pl.Float32,
    "cons.price.idx": pl.Float32,
    "cons.conf.idx": pl.Float32,
    "euribor3m": pl.Float32,
    "nr.employed": pl.Float32,
    "y": pl.Utf8,
}

factory = DataProviderFactory(kwargs=bankmark_common_args | {"name": "bankmark", "file_name": "bankmark.csv.gz", "sample_size": 1.0, "data_framework": "pandas"})
bankmark_pd = factory.create_data_provider()

factory = DataProviderFactory(kwargs=bankmark_common_args | {"name": "bankmark", "file_name": "bankmark.csv.gz", "sample_size": 1.0, "data_framework": "polars"})
bankmark_pl = factory.create_data_provider()

factory = DataProviderFactory(kwargs=bankmark_common_args | {"name": "bankmark_samp", "file_name": "bankmark_samp.csv.gz", "sample_size": 0.05, "data_framework": "pandas"})
bankmark_samp_pd = factory.create_data_provider()

factory = DataProviderFactory(kwargs=bankmark_common_args | {"name": "bankmark_samp", "file_name": "bankmark_samp.csv.gz", "sample_size": 0.05, "data_framework": "polars"})
bankmark_samp_pl = factory.create_data_provider()


breast_common_args = {
    "name": "breast",
    "file_name": "breast.csv.gz",
    "sample_size": 1.0,
    "schema": None,
    "class_col": "mb",
    "positive_class": "M",
    "spiel": """Creators:
    1. Dr. William H. Wolberg, General Surgery Dept.
    University of Wisconsin, Clinical Sciences Center
    Madison, WI 53792
    wolberg '@' eagle.surgery.wisc.edu

    2. W. Nick Street, Computer Sciences Dept.
    University of Wisconsin, 1210 West Dayton St., Madison, WI 53706
    street '@' cs.wisc.edu 608-262-6619

    3. Olvi L. Mangasarian, Computer Sciences Dept.
    University of Wisconsin, 1210 West Dayton St., Madison, WI 53706
    olvi '@' cs.wisc.edu

    Donor:
    Nick Street

    Data Set Information:
    Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image. A few of the images can be found at [Web Link]
    Separating plane described above was obtained using Multisurface Method-Tree (MSM-T) [K. P. Bennett, "Decision Tree Construction Via Linear Programming." Proceedings of the 4th Midwest Artificial Intelligence and Cognitive Science Society, pp. 97-101, 1992], a classification method which uses linear programming to construct a decision tree. Relevant features were selected using an exhaustive search in the space of 1-4 features and 1-3 separating planes.
    The actual linear program used to obtain the separating plane in the 3-dimensional space is that described in: [K. P. Bennett and O. L. Mangasarian: "Robust Linear Programming Discrimination of Two Linearly Inseparable Sets", Optimization Methods and Software 1, 1992, 23-34].

    This database is also available through the UW CS ftp server:
    ftp ftp.cs.wisc.edu
    cd math-prog/cpo-dataset/machine-learn/WDBC/

    Attribute Information:

    1) ID number
    2) Diagnosis (M = malignant, B = benign)
    3-32)
    """,
}

factory = DataProviderFactory(kwargs=breast_common_args | {"data_framework": "pandas"})
breast_pd = factory.create_data_provider()

factory = DataProviderFactory(kwargs=breast_common_args | {"data_framework": "polars"})
breast_pl = factory.create_data_provider()


car_common_args = {
    "name": "car",
    "file_name": "car.csv.gz",
    "sample_size": 1.0,
    "schema": None,
    "class_col": "acceptability",
    "positive_class": "acc",
    "spiel": """
    M. Bohanec and V. Rajkovic: Knowledge acquisition and explanation for
    multi-attribute decision making. In 8th Intl Workshop on Expert
    Systems and their Applications, Avignon, France. pages 59-78, 1988.

    Within machine-learning, this dataset was used for the evaluation
    of HINT (Hierarchy INduction Tool), which was proved to be able to
    completely reconstruct the original hierarchical model. This,
    together with a comparison with C4.5, is presented in

    B. Zupan, M. Bohanec, I. Bratko, J. Demsar: Machine learning by
    function decomposition. ICML-97, Nashville, TN. 1997 (to appear)
    """,
}

factory = DataProviderFactory(kwargs=car_common_args | {"data_framework": "pandas"})
car_pd = factory.create_data_provider()

factory = DataProviderFactory(kwargs=car_common_args | {"data_framework": "polars"})
car_pl = factory.create_data_provider()


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

factory = DataProviderFactory(kwargs=cardio_common_args | {"data_framework": "pandas"})
cardio_pd = factory.create_data_provider()

factory = DataProviderFactory(kwargs=cardio_common_args | {"data_framework": "polars"})
cardio_pl = factory.create_data_provider()


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
factory = DataProviderFactory(kwargs=cervical_common_args | h_common_args | {"data_framework": "pandas"})
cervicalh_pd = factory.create_data_provider()
cervicalh_pd.features.drop(columns=["Schiller", "Citology", "Biopsy"], axis=1, inplace=True)  # type: ignore
cervicalh_pd.spiel = f"This dataset uses `{h_common_args["class_col"]}` as the class column, removing the other three options\n" + cervicalh_pd.spiel

factory = DataProviderFactory(kwargs=cervical_common_args | h_common_args | {"data_framework": "polars"})
cervicalh_pl = factory.create_data_provider()
cervicalh_pl.features = cervicalh_pl.features.drop(["Schiller", "Citology", "Biopsy"])
cervicalh_pl.spiel = f"This dataset uses `{h_common_args["class_col"]}` as the class column, removing the other three options\n" + cervicalh_pl.spiel

s_common_args = {
    "name": "cervicals",
    "class_col": "Schiller",
}
factory = DataProviderFactory(kwargs=cervical_common_args | s_common_args | {"data_framework": "pandas"})
cervicals_pd = factory.create_data_provider()
cervicals_pd.features.drop(columns=["Hinselmann", "Citology", "Biopsy"], axis=1, inplace=True)  # type: ignore
cervicals_pd.spiel = f"This dataset uses `{s_common_args["class_col"]}` as the class column, removing the other three options\n" + cervicals_pd.spiel

factory = DataProviderFactory(kwargs=cervical_common_args | s_common_args | {"data_framework": "polars"})
cervicals_pl = factory.create_data_provider()
cervicals_pl.features = cervicals_pl.features.drop(["Hinselmann", "Citology", "Biopsy"])
cervicals_pl.spiel = f"This dataset uses `{s_common_args["class_col"]}` as the class column, removing the other three options\n" + cervicals_pl.spiel

c_common_args = {
    "name": "cervicalc",
    "class_col": "Citology",
}
factory = DataProviderFactory(kwargs=cervical_common_args | c_common_args | {"data_framework": "pandas"})
cervicalc_pd = factory.create_data_provider()
cervicalc_pd.features.drop(columns=["Hinselmann", "Schiller", "Biopsy"], axis=1, inplace=True)  # type: ignore
cervicalc_pd.spiel = f"This dataset uses `{c_common_args["class_col"]}` as the class column, removing the other three options\n" + cervicalc_pd.spiel

factory = DataProviderFactory(kwargs=cervical_common_args | c_common_args | {"data_framework": "polars"})
cervicalc_pl = factory.create_data_provider()
cervicalc_pl.features = cervicalc_pl.features.drop(["Hinselmann", "Schiller", "Biopsy"])
cervicalc_pl.spiel = f"This dataset uses `{c_common_args["class_col"]}` as the class column, removing the other three options\n" + cervicalc_pl.spiel

b_common_args = {
    "name": "cervicalb",
    "class_col": "Biopsy",
}
factory = DataProviderFactory(kwargs=cervical_common_args | b_common_args | {"data_framework": "pandas"})
cervicalb_pd = factory.create_data_provider()
cervicalb_pd.features.drop(columns=["Hinselmann", "Schiller", "Citology"], axis=1, inplace=True)  # type: ignore
cervicalb_pd.spiel = f"This dataset uses `{b_common_args["class_col"]}` as the class column, removing the other three options\n" + cervicalb_pd.spiel

factory = DataProviderFactory(kwargs=cervical_common_args | b_common_args | {"data_framework": "polars"})
cervicalb_pl = factory.create_data_provider()
cervicalb_pl.features = cervicalb_pl.features.drop(["Hinselmann", "Schiller", "Citology"])
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

factory = DataProviderFactory(kwargs=cervicalr_common_args | {"data_framework": "pandas"})
cervicalr_pd = factory.create_data_provider()

factory = DataProviderFactory(kwargs=cervicalr_common_args | {"data_framework": "polars"})
cervicalr_pl = factory.create_data_provider()


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

factory = DataProviderFactory(kwargs=credit_common_args | {"data_framework": "pandas"})
credit_pd = factory.create_data_provider()

factory = DataProviderFactory(kwargs=credit_common_args | {"data_framework": "polars"})
credit_pl = factory.create_data_provider()


diaretino_common_args = {
    "name": "diaretino",
    "file_name": "diaretino.csv.gz",
    "sample_size": 1.0,
    "schema": None,
    "class_col": "dr",
    "positive_class": "yes",
    "spiel": """Source:
    1. Dr. Balint Antal, Department of Computer Graphics and Image Processing
    Faculty of Informatics, University of Debrecen, 4010, Debrecen, POB 12, Hungary
    antal.balint '@' inf.unideb.hu
    2. Dr. Andras Hajdu, Department of Computer Graphics and Image Processing
    Faculty of Informatics, University of Debrecen, 4010, Debrecen, POB 12, Hungary
    hajdu.andras '@' inf.unideb.hu

    Data Set Information:
    This dataset contains features extracted from the Messidor image set to predict whether an image contains signs of diabetic retinopathy or not. All features represent either a detected lesion, a descriptive feature of a anatomical part or an image-level descriptor. The underlying method image analysis and feature extraction as well as our classification technique is described in Balint Antal, Andras Hajdu: An ensemble-based system for automatic screening of diabetic retinopathy, Knowledge-Based Systems 60 (April 2014), 20-27. The image set (Messidor) is available at [Web Link].

    Attribute Information:
    0) The binary result of quality assessment. 0 = bad quality 1 = sufficient quality.
    1) The binary result of pre-screening, where 1 indicates severe retinal abnormality and 0 its lack.
    2-7) The results of MA detection. Each feature value stand for the
    number of MAs found at the confidence levels alpha = 0.5, . . . , 1, respectively.
    8-15) contain the same information as 2-7) for exudates. However,
    as exudates are represented by a set of points rather than the number of
    pixels constructing the lesions, these features are normalized by dividing the
    number of lesions with the diameter of the ROI to compensate different image
    sizes.
    Note - 2-7 and 8-15 are not equal in number. Not sure what to call the last two ex.
    16) The euclidean distance of the center of
    the macula and the center of the optic disc to provide important information
    regarding the patientâ€™s condition. This feature
    is also normalized with the diameter of the ROI.
    17) The diameter of the optic disc.
    18) The binary result of the AM/FM-based classification.
    19) Class label. 1 = contains signs of DR (Accumulative label for the Messidor classes 1, 2, 3), 0 = no signs of DR.
    """,
}

factory = DataProviderFactory(kwargs=diaretino_common_args | {"data_framework": "pandas"})
diaretino_pd = factory.create_data_provider()

factory = DataProviderFactory(kwargs=diaretino_common_args | {"data_framework": "polars"})
diaretino_pl = factory.create_data_provider()


german_common_args = {
    "name": "german",
    "file_name": "german.csv.gz",
    "sample_size": 1.0,
    "schema": None,
    "class_col": "rating",
    "positive_class": "bad",
    "spiel": """Source:
    Professor Dr. Hans Hofmann
    Institut f"ur Statistik und "Okonometrie
    Universit"at Hamburg
    FB Wirtschaftswissenschaften
    Von-Melle-Park 5
    2000 Hamburg 13

    Data Set Information:
    Two datasets are provided. the original dataset, in the form provided by Prof. Hofmann, contains categorical/symbolic attributes and is in the file "german.data".
    For algorithms that need numerical attributes, Strathclyde University produced the file "german.data-numeric". This file has been edited and several indicator variables added to make it suitable for algorithms which cannot cope with categorical variables. Several attributes that are ordered categorical (such as attribute 17) have been coded as integer. This was the form used by StatLog.

    This dataset requires use of a cost matrix:
    . 1 2
    ------
    1 0 1
    -----
    2 5 0

    (1 = Good, 2 = Bad)
    The rows represent the actual classification and the columns the predicted classification.
    It is worse to class a customer as good when they are bad (5), than it is to class a customer as bad when they are good (1).
    """,
}

factory = DataProviderFactory(kwargs=german_common_args | {"data_framework": "pandas"})
german_pd = factory.create_data_provider()

factory = DataProviderFactory(kwargs=german_common_args | {"data_framework": "polars"})
german_pl = factory.create_data_provider()


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

factory = DataProviderFactory(kwargs=heart_common_args | {"data_framework": "pandas"})
heart_pd = factory.create_data_provider()

factory = DataProviderFactory(kwargs=heart_common_args | {"data_framework": "polars"})
heart_pl = factory.create_data_provider()


lending_common_args = {
    "class_col": "loan_status",
    "positive_class": "Fully Paid",
    "spiel": """Data Set Information:
    Originates from: https://www.lendingclub.com/info/download-data.action

    See also:
    https://www.kaggle.com/wordsforthewise/lending-club

    Prepared by Nate George:  https://github.com/nateGeorge/preprocess_lending_club_data
    """,
    "schema": None,
}

samp_common_args = {
    "name": "lending_samp",
    "file_name": "lending_samp.csv.gz",
    "sample_size": 0.1,
}

factory = DataProviderFactory(kwargs=lending_common_args | samp_common_args | {"data_framework": "pandas"})
lending_samp_pd = factory.create_data_provider()

factory = DataProviderFactory(kwargs=lending_common_args | samp_common_args | {"data_framework": "polars"})
lending_samp_pl = factory.create_data_provider()


small_samp_common_args = {
    "name": "lending_small_samp",
    "file_name": "lending_small_samp.csv.gz",
    "sample_size": 0.01,
}

factory = DataProviderFactory(kwargs=lending_common_args | small_samp_common_args | {"data_framework": "pandas"})
lending_small_samp_pd = factory.create_data_provider()

factory = DataProviderFactory(kwargs=lending_common_args | small_samp_common_args | {"data_framework": "polars"})
lending_small_samp_pl = factory.create_data_provider()


tiny_samp_common_args = {
    "name": "lending_tiny_samp",
    "file_name": "lending_tiny_samp.csv.gz",
    "sample_size": 0.0025,
}

factory = DataProviderFactory(kwargs=lending_common_args | tiny_samp_common_args | {"data_framework": "pandas"})
lending_tiny_samp_pd = factory.create_data_provider()

factory = DataProviderFactory(kwargs=lending_common_args | tiny_samp_common_args | {"data_framework": "polars"})
lending_tiny_samp_pl = factory.create_data_provider()


mhtech14_common_args = {
    "name": "mhtech14",
    "file_name": "mhtech14.csv.gz",
    "class_col": "treatment",
    "positive_class": "Yes",
    "sample_size": 1.0,
    "spiel": """
    From Kaggle - https://www.kaggle.com/osmi/mental-health-in-tech-survey

    This dataset contains the following data:

    Timestamp
    Age
    Gender
    Country
    state: If you live in the United States, which state or territory do you live in?
    self_employed: Are you self-employed?
    family_history: Do you have a family history of mental illness?
    treatment: Have you sought treatment for a mental health condition?
    work_interfere: If you have a mental health condition, do you feel that it interferes with your work?
    no_employees: How many employees does your company or organization have?
    remote_work: Do you work remotely (outside of an office) at least 50% of the time?
    tech_company: Is your employer primarily a tech company/organization?
    benefits: Does your employer provide mental health benefits?
    care_options: Do you know the options for mental health care your employer provides?
    wellness_program: Has your employer ever discussed mental health as part of an employee wellness program?
    seek_help: Does your employer provide resources to learn more about mental health issues and how to seek help?
    anonymity: Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources?
    leave: How easy is it for you to take medical leave for a mental health condition?
    mental_health_consequence: Do you think that discussing a mental health issue with your employer would have negative consequences?
    phys_health_consequence: Do you think that discussing a physical health issue with your employer would have negative consequences?
    coworkers: Would you be willing to discuss a mental health issue with your coworkers?
    supervisor: Would you be willing to discuss a mental health issue with your direct supervisor(s)?
    mental_health_interview: Would you bring up a mental health issue with a potential employer in an interview?
    phys_health_interview: Would you bring up a physical health issue with a potential employer in an interview?
    mental_vs_physical: Do you feel that your employer takes mental health as seriously as physical health?
    obs_consequence: Have you heard of or observed negative consequences for coworkers with mental health conditions in your workplace?
    comments: Any additional notes or comments
    """,
    "schema": None,
}

factory = DataProviderFactory(kwargs=mhtech14_common_args | {"data_framework": "pandas"})
mhtech14_pd = factory.create_data_provider()
mhtech14_pd.features.drop(columns=["comments"], axis=1, inplace=True)  # type: ignore

factory = DataProviderFactory(kwargs=mhtech14_common_args | {"data_framework": "polars"})
mhtech14_pl = factory.create_data_provider()
mhtech14_pl.features = mhtech14_pl.features.drop(["comments"])


mhtech16_common_args = {
    "file_name": "mhtech16.csv.gz",
    "positive_class": "yes",
    "spiel": """
    From Kaggle. The three columns used for treatment are as follows and have been duplicated with shorter keys to make pre-processing easier:
    mh1 = 'Have you ever sought treatment for a mental health issue from a mental health professional?'
    mh2 = 'Have you been diagnosed with a mental health condition by a medical professional?'
    mh3 = 'Do you currently have a mental health disorder?'

    There is also corruption in the file, with the column 'Why or why not?' being duplicated as 'Why or why not?.1', which is somewhat buggy to remove.

    These issues have been addressed in the pre-processing of the data.
    """,
    "sample_size": 1.0,
    "schema": None,
}


def preproc_extra(data_container: DataProvider) -> DataProvider:
    treatment_columns = ["mh1", "mh2", "mh3"]
    treatment_columns.remove(data_container.class_col)

    drop_columns = [
        "Have you ever sought treatment for a mental health issue from a mental health professional?",
        "Have you been diagnosed with a mental health condition by a medical professional?",
        "Do you currently have a mental health disorder?",
        "If you have revealed a mental health issue to a client or business contact - do you believe this has impacted you negatively?",
        "Was your anonymity protected if you chose to take advantage of mental health or substance abuse treatment resources with previous employers?",
        "If you have been diagnosed or treated for a mental health disorder - do you ever reveal this to coworkers or employees?",
        "If you have revealed a mental health issue to a coworker or employee - do you believe this has impacted you negatively?",
        "Do you believe your productivity is ever affected by a mental health issue?",
        "If yes - what percentage of your work time (time performing primary or secondary job functions) is affected by a mental health issue?",
        "If you have a mental health issue - do you feel that it interferes with your work when being treated effectively?",
        "If you have a mental health issue - do you feel that it interferes with your work when NOT being treated effectively?",
        "How willing would you be to share with friends and family that you have a mental illness?",
        "If yes - what condition(s) have you been diagnosed with?",
        "If maybe - what condition(s) do you believe you have?",
        "If so - what condition(s) were you diagnosed with?",
    ]
    corrupted_col = "Why or why not?"
    if isinstance(data_container.features, pd.DataFrame):
        data_container.features.drop(columns=drop_columns + treatment_columns, axis=1, inplace=True)
        for c in data_container.features.columns:
            if corrupted_col in c:
                data_container.features.drop(columns=c, axis=1, inplace=True)
    elif isinstance(data_container.features, pl.DataFrame):
        data_container.features = data_container.features.drop(drop_columns + treatment_columns)
        for c in data_container.features.columns:
            if corrupted_col in c:
                data_container.features = data_container.features.drop(c)
    data_container.spiel = f"This dataset uses '{data_container.class_col}' as the class column, removing the other two options\n" + data_container.spiel
    return data_container


mh1_common_args = {
    "name": "mh1tech16",
    "class_col": "mh1",
}
factory = DataProviderFactory(kwargs=mhtech16_common_args | mh1_common_args | {"data_framework": "pandas"})
mh1tech16_pd = factory.create_data_provider()
mh1tech16_pd = preproc_extra(mh1tech16_pd)

factory = DataProviderFactory(kwargs=mhtech16_common_args | mh1_common_args | {"data_framework": "polars"})
mh1tech16_pl = factory.create_data_provider()
mh1tech16_pl = preproc_extra(mh1tech16_pl)

mh2_common_args = {
    "name": "mh2tech16",
    "class_col": "mh2",
}
factory = DataProviderFactory(kwargs=mhtech16_common_args | mh2_common_args | {"data_framework": "pandas"})
mh2tech16_pd = factory.create_data_provider()
mh2tech16_pd = preproc_extra(mh2tech16_pd)

factory = DataProviderFactory(kwargs=mhtech16_common_args | mh2_common_args | {"data_framework": "polars"})
mh2tech16_pl = factory.create_data_provider()
mh2tech16_pl = preproc_extra(mh2tech16_pl)

mh3_common_args = {
    "name": "mh3tech16",
    "class_col": "mh3",
}
factory = DataProviderFactory(kwargs=mhtech16_common_args | mh3_common_args | {"data_framework": "pandas"})
mh3tech16_pd = factory.create_data_provider()
mh3tech16_pd = preproc_extra(mh3tech16_pd)

factory = DataProviderFactory(kwargs=mhtech16_common_args | mh3_common_args | {"data_framework": "polars"})
mh3tech16_pl = factory.create_data_provider()
mh3tech16_pl = preproc_extra(mh3tech16_pl)

mush_common_args = {
    "name": "mush",
    "file_name": "mush.csv.gz",
    "sample_size": 1.0,
    "schema": None,
    "class_col": "edible",
    "positive_class": "e",
    "spiel": """Source:
    This dataset was taken from the UCI Machine Learning Repository. The data was donated by Jeff Schlimmer.

    Data Set Information:
    This data set includes descriptions of hypothetical samples corresponding to 23 species of gilled mushrooms in the Agaricus and Lepiota Family (pp. 500-525). Each species is identified as definitely edible, definitely poisonous, or of unknown edibility and not recommended. This latter class was combined with the poisonous one. The Guide clearly states that there is no simple rule for determining the edibility of a mushroom; no rule like ``leaflets three, let it be'' for Poisonous Oak and Ivy.

    Attribute Information:
    1. cshape (cap-shape): bell=b, conical=c, convex=x, flat=f, knobbed=k, sunken=s
    2. csurface (cap-surface): fibrous=f, grooves=g, scaly=y, smooth=s
    3. ccolor (cap-color): brown=n, buff=b, cinnamon=c, gray=g, green=r, pink=p, purple=u, red=e, white=w, yellow=y
    4. bruises: bruises=t, no=f
    5. odor: almond=a, anise=l, creosote=c, fishy=y, foul=f, musty=m, none=n, pungent=p, spicy=s
    6. gattach (gill-attachment): attached=a, descending=d, free=f, notched=n
    7. gspace (gill-spacing): close=c, crowded=w, distant=d
    8. gsize (gill-size): broad=b, narrow=n
    9. gcolor (gill-color): black=k, brown=n, buff=b, chocolate=h, gray=g, green=r, orange=o, pink=p, purple=u, red=e, white=w, yellow=y
    10. shape (stalk-shape): enlarging=e, tapering=t
    11. sroot (stalk-root): bulbous=b, club=c, cup=u, equal=e, rhizomorphs=z, rooted=r, missing=?
    12. ssurfaring (stalk-surface-above-ring): fibrous=f, scaly=y, silky=k, smooth=s
    13. ssurfbring (stalk-surface-below-ring): fibrous=f, scaly=y, silky=k, smooth=s
    14. scoloraring (stalk-color-above-ring): brown=n, buff=b, cinnamon=c, gray=g, orange=o, pink=p, red=e, white=w, yellow=y
    15. scolorbring (stalk-color-below-ring): brown=n, buff=b, cinnamon=c, gray=g, orange=o, pink=p, red=e, white=w, yellow=y
    16. vtype (veil-type): partial=p, universal=u
    17. vcolor (veil-color): brown=n, orange=o, white=w, yellow=y
    18. rnum (ring-number): none=n, one=o,
    19. type (ring-type): cobwebby=c, evanescent=e, flaring=f, large=l, none=n, pendant=p, sheathing=s, zone=z
    20. sporecolor (spore-print-color): black=k, brown=n, buff=b, chocolate=h, green=r, orange=o, purple=u, white=w, yellow=y
    21. pop (population): abundant=a, clustered=c, numerous=n, scattered=s, several=v, solitary=y
    22. hab (habitat): grasses=g, leaves=l, meadows=m, paths=p, urban=u, waste=w, woods=d
    """,
}

factory = DataProviderFactory(kwargs=mush_common_args | {"data_framework": "pandas"})
mush_pd = factory.create_data_provider()

factory = DataProviderFactory(kwargs=mush_common_args | {"data_framework": "polars"})
mush_pl = factory.create_data_provider()


noshow_common_args = {
    "name": "noshow",
    "file_name": "noshow.csv.gz",
    "sample_size": 1.0,
    "schema": None,
    "class_col": "no_show",
    "positive_class": "Yes",
    "spiel": """Source:
    No further information
    """,
}

noshow_full_args = {"name": "noshow", "sample_size": 1.0, "file_name": "noshow.csv.gz"}

factory = DataProviderFactory(kwargs=noshow_common_args | noshow_full_args | {"data_framework": "pandas"})
noshow_pd = factory.create_data_provider()

factory = DataProviderFactory(kwargs=noshow_common_args | noshow_full_args | {"data_framework": "polars"})
noshow_pl = factory.create_data_provider()

noshow_samp_args = {"name": "noshow_samp", "sample_size": 0.2, "file_name": "noshow_samp.csv.gz"}

factory = DataProviderFactory(kwargs=noshow_common_args | noshow_samp_args | {"data_framework": "pandas"})
noshow_samp_pd = factory.create_data_provider()

factory = DataProviderFactory(kwargs=noshow_common_args | noshow_samp_args | {"data_framework": "polars"})
noshow_samp_pl = factory.create_data_provider()

noshow_small_samp_args = {"name": "noshow_small_samp", "sample_size": 0.02, "file_name": "noshow_small_samp.csv.gz"}

factory = DataProviderFactory(kwargs=noshow_common_args | noshow_small_samp_args | {"data_framework": "pandas"})
noshow_small_samp_pd = factory.create_data_provider()

factory = DataProviderFactory(kwargs=noshow_common_args | noshow_small_samp_args | {"data_framework": "polars"})
noshow_small_samp_pl = factory.create_data_provider()


nursery_polars_schema = {
    "parents": pl.Categorical,
    "has_nurs": pl.Categorical,
    "form": pl.Categorical,
    "children": pl.Categorical,
    "housing": pl.Categorical,
    "finance": pl.Categorical,
    "social": pl.Categorical,
    "health": pl.Categorical,
    "decision": pl.Categorical,
}


nursery_common_args = {
    "name": "nursery",

    "schema": None,
    "class_col": "decision",
    "positive_class": None,
    "spiel": """Data Description:
    The target is a multinomial response variable.
    Nursery Database was derived from a hierarchical decision model
    originally developed to rank applications for nursery schools. It
    was used during several years in 1980's when there was excessive
    enrollment to these schools in Ljubljana, Slovenia, and the
    rejected applications frequently needed an objective
    explanation. The final decision depended on three subproblems:
    occupation of parents and child's nursery, family structure and
    financial standing, and social and health picture of the family.
    The model was developed within expert system shell for decision
    making DEX (M. Bohanec, V. Rajkovic: Expert system for decision
    making. Sistemica 1(1), pp. 145-157, 1990.).
    """,
}

full_nursery_args = {"name": "nursery", "sample_size": 1.0, "file_name": "nursery.csv.gz"}
samp_nursery_args = {"name": "nursery_samp", "sample_size": 0.2, "file_name": "nursery_samp.csv.gz"}

factory = DataProviderFactory(kwargs=nursery_common_args | full_nursery_args | {"data_framework": "pandas", "schema": None})
nursery_pd = factory.create_data_provider()

factory = DataProviderFactory(kwargs=nursery_common_args | full_nursery_args | {"data_framework": "polars", "schema": nursery_polars_schema})
nursery_pl = factory.create_data_provider()

factory = DataProviderFactory(kwargs=nursery_common_args | samp_nursery_args | {"data_framework": "pandas", "schema": None})
nursery_samp_pd = factory.create_data_provider()

factory = DataProviderFactory(kwargs=nursery_common_args | samp_nursery_args | {"data_framework": "polars", "schema": nursery_polars_schema})
nursery_samp_pl = factory.create_data_provider()

rcdv_common_args = {
    "schema": None,
    "class_col": "recid",
    "positive_class": "Y",
    "spiel": """
    Data Set Information:
    This is a description of the data on the file, DATA1978.
    The description was prepared by Peter Schmidt, Department of Economics, Michigan State University, East Lansing, Michigan 48824.
    The data were gathered as part of a grant from the National Institute of Justice to Peter Schmidt and Ann Witte, “Improving Predictions of Recidivism by Use of Individual Characteristics,” 84-IJ-CX-0021.
    A more complete description of the data, and of the uses to which they were put, can be found in the final report for this grant.
    Another similar dataset, contained in a file DATA1980 on a separate diskette, is also described in that report.

    The North Carolina Department of Correction furnished a data tape which was to contain information on all individuals released from a North Carolina prison during the period from July 1, 1977 through June 30, 1978.
    There were 9457 individual records on this tape. However, 130 records were deleted because of obvious defects.
    In almost all cases, the reason for deletion is that the individual’s date of release was in fact not during the time period which defined the data set.
    This left a total of 9327 individual records, and accordingly there are 9327 records on DATA1978.

    The basic sample of 9327 observations contained many observations for which one or more of the variables used in our analyses were missing.
    Specifically, 4709 observations were missing information on one or more such variables, and these 4709 observations constitute the “missing data” file.
    The other 4618 observations which contained complete information were randomly split into an “analysis file” of 1540 observations and a “validation file” of 3078 observations.

    DATA 1978 contains 9327 individual records. Each individual record contains 28 columns of data, representing the following 19 variables.

    WHITE ALCHY JUNKY SUPER MARRIED FELON WORKREL PROPTY PERSON
    1 2 3 4 5 6 7 8 9 10 11-12 13-14 15-16 17-19 20-22 23-24 25-27 28

    WHITE is a dummy (indicator) variable equal to zero if the individual is black, and equal to one otherwise. Basically, WHITE equals one for whites and zero for blacks. However, the North Carolina prison population also contains a small number of Native Americans, Hispanics, Orientals, and individuals of “other” race. They are treated as whites, by the above definition.
    ALCHY is a dummy variable equal to one if the individual’s record indicates a serious problem with alcohol, and equal to zero otherwise. It is important to note that for individuals in the missing data sample (FILE = 3), the value of ALCHY is recorded as zero, but is meaningless.
    JUNKY is a dummy variable equal to one if the individual’s record indicates use of hard drugs, and equal to zero otherwise. It is important to note that for individuals in the missing data sample (FILE = 3), the value of JUNKY is recorded as zero, but is meaningless.
    SUPER is a dummy variable equal to one if the individual’s release from the sample sentence was supervised (e.g., parole), and equal to zero otherwise.
    MARRIED is a dummy variable equal to one if the individual was married at the time of release from the sample sentence, and equal to zero otherwise.
    FELON is a dummy variable equal to one if the sample conviction was for a felony, and equal to zero if it was for a misdemeanor.
    WORKREL is a dummy variable equal to one if the individual participated in the North Carolina prisoner work release program during the sample sentence, and equal to zero otherwise.
    PROPTY is a dummy variable equal to one if the sample conviction was for a crime against property, and equal to zero otherwise. A detailed listing of the crime codes which define this variable (and PERSON below) can be found in A. Witte, Work Release in North Carolina: An Evaluation of Its Post Release Effects, Chapel Hill, North Carolina: Institute for Research in Social Science.
    PERSON is a dummy variable equal to one if the sample conviction was for a crime against a person, and equal to zero otherwise. (Incidentally, note that PROPTY plus PERSON is not necessarily equal to one, because there is an additional miscellaneous category of offenses which are neither offenses against property nor offenses against a person.)
    MALE is a dummy variable equal to one if the individual is male, and equal to zero if the individual is female.
    PRIORS is the number of previous incarcerations, not including the sample sentence. The value -9 indicates that this information is missing.
    SCHOOL is the number of years of formal schooling completed. The value zero indicates that this information is missing.
    RULE is the number of prison rule violations reported during the sample sentence.
    AGE is age (in months) at time of release.
    TSERVD is the time served (in months) for the sample sentence.
    FOLLOW is the length of the followup period, in months. (The followup period is the time from relase until the North Carolina Department of Correction records were searched, in April, 1984.)
    RECID is a dummy variable equal to one if the individual returned to a North Carolina prison during the followup period, and equal to zero otherwise.
    TIME is the length of time from release from the sample sentence until return to prison in North Carolina, for individuals for whom RECID equals one. TIME is rounded to the nearest month. (In particular, note that TIME equals zero for individuals who return to prison in North Carolina within the first half month after release.) For individuals for whom RECID equals zero, the value of TIME is meaningless. For such individuals, TIME is usually recorded as zero, but it is occasionally recorded as the length of the followup period. We emphasize again that neither value is meaningful, for those individuals for whom RECID equals zero.
    FILE is a variable indicating to which data sample the individual record belongs. The value 1 indicates the analysis sample, 2 the validation sampel and 3 is missing data sample.
    """,
}

rcdv_full_args = {"name": "rcdv", "sample_size": 1.0, "file_name": "rcdv.csv.gz"}
rcdv_samp_args = {"name": "rcdv_samp", "sample_size": 0.1, "file_name": "rcdv_samp.csv.gz"}

factory = DataProviderFactory(kwargs=rcdv_common_args | rcdv_full_args | {"data_framework": "pandas"})
rcdv_pd = factory.create_data_provider()

factory = DataProviderFactory(kwargs=rcdv_common_args | rcdv_full_args | {"data_framework": "polars"})
rcdv_pl = factory.create_data_provider()

factory = DataProviderFactory(kwargs=rcdv_common_args | rcdv_samp_args | {"data_framework": "pandas"})
rcdv_samp_pd = factory.create_data_provider()
rcdv_samp_pd.spiel = "A random 0.1% sample without replacement of the original rcdv dataset.\n" + rcdv_samp_pd.spiel

factory = DataProviderFactory(kwargs=rcdv_common_args | rcdv_samp_args | {"data_framework": "polars"})
rcdv_samp_pl = factory.create_data_provider()
rcdv_samp_pl.spiel = "A random 0.1% sample without replacement of the original rcdv dataset.\n" + rcdv_samp_pl.spiel


readmit_common_args = {
    "class_col": "readmitted",
    "positive_class": "T",
    "spiel": """
    From Kaggle - https://www.kaggle.com/dansbecker/hospital-readmissions
    No further information
    """,
}

readmit_full_args = {"name": "readmit", "sample_size": 1.0, "file_name": "readmit.csv.gz"}
readmit_samp_args = {"name": "readmit_samp", "sample_size": 0.1, "file_name": "readmit_samp.csv.gz"}

factory = DataProviderFactory(kwargs=readmit_common_args | readmit_full_args | {"data_framework": "pandas"})
readmit_pd = factory.create_data_provider()

factory = DataProviderFactory(kwargs=readmit_common_args | readmit_full_args | {"data_framework": "polars"})
readmit_pl = factory.create_data_provider()

factory = DataProviderFactory(kwargs=readmit_common_args | readmit_samp_args | {"data_framework": "pandas"})
readmit_samp_pd = factory.create_data_provider()

factory = DataProviderFactory(kwargs=readmit_common_args | readmit_samp_args | {"data_framework": "polars"})
readmit_samp_pl = factory.create_data_provider()


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

factory = DataProviderFactory(kwargs=thyroid_common_args | thyroid_full_args | {"data_framework": "pandas"})
thyroid_pd = factory.create_data_provider()

factory = DataProviderFactory(kwargs=thyroid_common_args | thyroid_full_args | {"data_framework": "polars"})
thyroid_pl = factory.create_data_provider()

factory = DataProviderFactory(kwargs=thyroid_common_args | thyroid_samp_args | {"data_framework": "pandas"})
thyroid_samp_pd = factory.create_data_provider()

factory = DataProviderFactory(kwargs=thyroid_common_args | thyroid_samp_args | {"data_framework": "polars"})
thyroid_samp_pl = factory.create_data_provider()
