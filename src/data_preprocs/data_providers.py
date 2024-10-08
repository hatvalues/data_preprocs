from .data_loading import create_data_provider
import polars as pl


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
}

adult_sample_sizes = {
    "full": 1.0,
    "samp": 0.25,
    "small_samp": 0.025,
}

adult_pd = create_data_provider(
    name="adult",
    file_name="adult.csv.gz",
    sample_size=1.0,
    data_framework="pandas",
    schema=None,
    **adult_common_args,
)

adult_samp_pd = create_data_provider(
    name="adult_samp",
    file_name="adult_samp.csv.gz",
    sample_size=0.25,
    data_framework="pandas",
    schema=None,
    **adult_common_args,
)

adult_small_samp_pd = create_data_provider(
    name="adult_small_samp",
    file_name="adult_small_samp.csv.gz",
    sample_size=0.025,
    data_framework="pandas",
    schema=None,
    **adult_common_args,
)


adult_pl = create_data_provider(
    name="adult",
    file_name="adult.csv.gz",
    sample_size=1.0,
    data_framework="polars",
    schema=None,
    **adult_common_args,
)


adult_samp_pl = create_data_provider(
    name="adult_samp",
    file_name="adult_samp.csv.gz",
    sample_size=0.25,
    data_framework="polars",
    schema=None,
    **adult_common_args,
)

adult_small_samp_pl = create_data_provider(
    name="adult_small_samp",
    file_name="adult_small_samp.csv.gz",
    sample_size=0.025,
    data_framework="polars",
    schema=None,
    **adult_common_args,
)


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


bankmark_pd = create_data_provider(
    name="bankmark",
    file_name="bankmark.csv.gz",
    sample_size=1.0,
    schema=bankmark_pandas_schema,
    **bankmark_common_args,
)


bankmark_samp_pd = create_data_provider(
    name="bankmark_samp",
    file_name="bankmark_samp.csv.gz",
    sample_size=0.05,
    schema=bankmark_pandas_schema,
    **bankmark_common_args,
)

bankmark_pl = create_data_provider(
    name="bankmark",
    file_name="bankmark.csv.gz",
    sample_size=1.0,
    schema=bankmark_polars_schema,
    **bankmark_common_args,
)


bankmark_samp_pl = create_data_provider(
    name="bankmark_samp",
    file_name="bankmark_samp.csv.gz",
    sample_size=0.05,
    schema=bankmark_polars_schema,
    **bankmark_common_args,
)


breast_common_args = {
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

breast_pd = create_data_provider(
    name="breast",
    file_name="breast.csv.gz",
    sample_size=1.0,
    data_framework="pandas",
    schema=None,
    **breast_common_args,
)

breast_pl = create_data_provider(
    name="breast",
    file_name="breast.csv.gz",
    sample_size=1.0,
    data_framework="polars",
    schema=None,
    **breast_common_args,
)


car_common_args = {
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


car_pd = create_data_provider(
    name="car",
    file_name="car.csv.gz",
    sample_size=1.0,
    data_framework="pandas",
    schema=None,
    **car_common_args,
)


car_pl = create_data_provider(
    name="car",
    file_name="car.csv.gz",
    sample_size=1.0,
    data_framework="polars",
    schema=None,
    **car_common_args,
)


cardio_common_args = {
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

cardio_pd = create_data_provider(
    name="cardio",
    file_name="cardio.csv.gz",
    sample_size=1.0,
    data_framework="pandas",
    schema=None,
    **cardio_common_args,
)


cardio_pl = create_data_provider(
    name="cardio",
    file_name="cardio.csv.gz",
    sample_size=1.0,
    data_framework="polars",
    schema=None,
    **cardio_common_args,
)


cervical_common_args = {
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

cervicalh_pd = create_data_provider(
    name="cervicalh",
    file_name="cervical.csv.gz",
    sample_size=1.0,
    data_framework="pandas",
    schema=None,
    class_col="Hinselmann",
    positive_class="T",
    **cervical_common_args,
)

cervicalh_pd.features.drop(columns=["Schiller", "Citology", "Biopsy"], axis=1, inplace=True)  # type: ignore
cervicalh_pd.spiel = "This dataset uses 'Hinselmann' as the class column, removing the other three options\n" + cervicalh_pd.spiel

cervicalh_pl = create_data_provider(
    name="cervicalh",
    file_name="cervical.csv.gz",
    sample_size=1.0,
    data_framework="polars",
    schema=None,
    class_col="Hinselmann",
    positive_class="T",
    **cervical_common_args,
)

cervicalh_pl.features = cervicalh_pl.features.drop(["Schiller", "Citology", "Biopsy"])
cervicalh_pl.spiel = "This dataset uses 'Hinselmann' as the class column, removing the other three options\n" + cervicalh_pl.spiel

cervicals_pd = create_data_provider(
    name="cervicals",
    file_name="cervical.csv.gz",
    sample_size=1.0,
    data_framework="pandas",
    schema=None,
    class_col="Schiller",
    positive_class="T",
    **cervical_common_args,
)

cervicals_pd.features.drop(columns=["Hinselmann", "Citology", "Biopsy"], axis=1, inplace=True)  # type: ignore
cervicals_pd.spiel = "This dataset uses 'Schiller' as the class column, removing the other three options\n" + cervicals_pd.spiel

cervicals_pl = create_data_provider(
    name="cervicals",
    file_name="cervical.csv.gz",
    sample_size=1.0,
    data_framework="polars",
    schema=None,
    class_col="Schiller",
    positive_class="T",
    **cervical_common_args,
)

cervicals_pl.features = cervicals_pl.features.drop(["Hinselmann", "Citology", "Biopsy"])
cervicals_pl.spiel = "This dataset uses 'Schiller' as the class column, removing the other three options\n" + cervicals_pl.spiel

cervicalc_pd = create_data_provider(
    name="cervicalc",
    file_name="cervical.csv.gz",
    sample_size=1.0,
    data_framework="pandas",
    schema=None,
    class_col="Citology",
    positive_class="T",
    **cervical_common_args,
)

cervicalc_pd.features.drop(columns=["Hinselmann", "Schiller", "Biopsy"], axis=1, inplace=True)  # type: ignore
cervicalc_pd.spiel = "This dataset uses 'Citology' as the class column, removing the other three options\n" + cervicalc_pd.spiel

cervicalc_pl = create_data_provider(
    name="cervicalc",
    file_name="cervical.csv.gz",
    sample_size=1.0,
    data_framework="polars",
    schema=None,
    class_col="Citology",
    positive_class="T",
    **cervical_common_args,
)

cervicalc_pl.features = cervicalc_pl.features.drop(["Hinselmann", "Schiller", "Biopsy"])
cervicalc_pl.spiel = "This dataset uses 'Citology' as the class column, removing the other three options\n" + cervicalc_pl.spiel

cervicalb_pd = create_data_provider(
    name="cervicalb",
    file_name="cervical.csv.gz",
    sample_size=1.0,
    data_framework="pandas",
    schema=None,
    class_col="Biopsy",
    positive_class="T",
    **cervical_common_args,
)

cervicalb_pd.features.drop(columns=["Hinselmann", "Schiller", "Citology"], axis=1, inplace=True)  # type: ignore
cervicalb_pd.spiel = "This dataset uses 'Biopsy' as the class column, removing the other three options\n" + cervicalb_pd.spiel

cervicalb_pl = create_data_provider(
    name="cervicalb",
    file_name="cervical.csv.gz",
    sample_size=1.0,
    data_framework="polars",
    schema=None,
    class_col="Biopsy",
    positive_class="T",
    **cervical_common_args,
)

cervicalb_pl.features = cervicalb_pl.features.drop(["Hinselmann", "Schiller", "Citology"])
cervicalb_pl.spiel = "This dataset uses 'Biopsy' as the class column, removing the other three options\n" + cervicalb_pl.spiel


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
}

cervicalr_pd = create_data_provider(
    data_framework="pandas",
    schema=None,
    **cervicalr_common_args,  # type: ignore
)

cervicalr_pl = create_data_provider(
    data_framework="polars",
    schema=None,
    **cervicalr_common_args,  # type: ignore
)


credit_common_args = {
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

credit_pd = create_data_provider(
    name="credit",
    file_name="credit.csv.gz",
    sample_size=1.0,
    data_framework="pandas",
    schema=None,
    **credit_common_args,
)

credit_pl = create_data_provider(
    name="credit",
    file_name="credit.csv.gz",
    sample_size=1.0,
    data_framework="polars",
    schema=None,
    **credit_common_args,
)


diaretino_common_args = {
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

diaretino_pd = create_data_provider(
    name="diartino",
    file_name="diaretino.csv.gz",
    sample_size=1.0,
    data_framework="pandas",
    schema=None,
    **diaretino_common_args,
)

diaretino_pl = create_data_provider(
    name="diaretino",
    file_name="diaretino.csv.gz",
    sample_size=1.0,
    data_framework="polars",
    schema=None,
    **diaretino_common_args,
)


german_common_args = {
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

german_pd = create_data_provider(
    name="german",
    file_name="german.csv.gz",
    sample_size=1.0,
    data_framework="pandas",
    schema=None,
    **german_common_args,
)

german_pl = create_data_provider(
    name="german",
    file_name="german.csv.gz",
    sample_size=1.0,
    data_framework="polars",
    schema=None,
    **german_common_args,
)


heart_common_args = {
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

heart_pd = create_data_provider(
    name="heart",
    file_name="heart.csv.gz",
    sample_size=1.0,
    data_framework="pandas",
    schema=None,
    **heart_common_args,
)

heart_pl = create_data_provider(
    name="heart",
    file_name="heart.csv.gz",
    sample_size=1.0,
    data_framework="polars",
    schema=None,
    **heart_common_args,
)


lending_common_args = {
    "class_col": "loan_status",
    "positive_class": "Fully Paid",
    "spiel": """Data Set Information:
    Originates from: https://www.lendingclub.com/info/download-data.action

    See also:
    https://www.kaggle.com/wordsforthewise/lending-club

    Prepared by Nate George:  https://github.com/nateGeorge/preprocess_lending_club_data
    """,
}

lending_samp_pd = create_data_provider(
    name="lending_samp",
    file_name="lending_samp.csv.gz",
    sample_size=0.1,
    data_framework="pandas",
    schema=None,
    **lending_common_args,
)

lending_samp_pl = create_data_provider(
    name="lending_samp",
    file_name="lending_samp.csv.gz",
    sample_size=0.1,
    data_framework="polars",
    schema=None,
    **lending_common_args,
)

lending_small_samp_pd = create_data_provider(
    name="lending_small_samp",
    file_name="lending_small_samp.csv.gz",
    sample_size=0.01,
    data_framework="pandas",
    schema=None,
    **lending_common_args,
)

lending_small_samp_pl = create_data_provider(
    name="lending_small_samp",
    file_name="lending_small_samp.csv.gz",
    sample_size=0.01,
    data_framework="polars",
    schema=None,
    **lending_common_args,
)

lending_tiny_samp_pd = create_data_provider(
    name="lending_tiny_samp",
    file_name="lending_tiny_samp.csv.gz",
    sample_size=0.0025,
    data_framework="pandas",
    schema=None,
    **lending_common_args,
)

lending_tiny_samp_pl = create_data_provider(
    name="lending_tiny_samp",
    file_name="lending_tiny_samp.csv.gz",
    sample_size=0.0025,
    data_framework="polars",
    schema=None,
    **lending_common_args,
)


# # mental health survey 2014
# mhtech14 = dict(
#     dataset_name = 'mhtech14',
#     drop = 'comments',
#     class_col = 'treatment',

#     positive_class='Yes',
#     spiel = '''
#     From Kaggle - https://www.kaggle.com/osmi/mental-health-in-tech-survey

#     This dataset contains the following data:

#     Timestamp
#     Age
#     Gender
#     Country
#     state: If you live in the United States, which state or territory do you live in?
#     self_employed: Are you self-employed?
#     family_history: Do you have a family history of mental illness?
#     treatment: Have you sought treatment for a mental health condition?
#     work_interfere: If you have a mental health condition, do you feel that it interferes with your work?
#     no_employees: How many employees does your company or organization have?
#     remote_work: Do you work remotely (outside of an office) at least 50% of the time?
#     tech_company: Is your employer primarily a tech company/organization?
#     benefits: Does your employer provide mental health benefits?
#     care_options: Do you know the options for mental health care your employer provides?
#     wellness_program: Has your employer ever discussed mental health as part of an employee wellness program?
#     seek_help: Does your employer provide resources to learn more about mental health issues and how to seek help?
#     anonymity: Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources?
#     leave: How easy is it for you to take medical leave for a mental health condition?
#     mental_health_consequence: Do you think that discussing a mental health issue with your employer would have negative consequences?
#     phys_health_consequence: Do you think that discussing a physical health issue with your employer would have negative consequences?
#     coworkers: Would you be willing to discuss a mental health issue with your coworkers?
#     supervisor: Would you be willing to discuss a mental health issue with your direct supervisor(s)?
#     mental_health_interview: Would you bring up a mental health issue with a potential employer in an interview?
#     phys_health_interview: Would you bring up a physical health issue with a potential employer in an interview?
#     mental_vs_physical: Do you feel that your employer takes mental health as seriously as physical health?
#     obs_consequence: Have you heard of or observed negative consequences for coworkers with mental health conditions in your workplace?
#     comments: Any additional notes or comments
#     '''

# this as a pandas schema (dctionary of types) and then as a polars schema
#     Age
#     Gender
#     Country
#     state: If you live in the United States, which state or territory do you live in?
#     self_employed: Are you self-employed?
#     family_history: Do you have a family history of mental illness?
#     treatment: Have you sought treatment for a mental health condition?
#     work_interfere: If you have a mental health condition, do you feel that it interferes with your work?
#     no_employees: How many employees does your company or organization have?
#     remote_work: Do you work remotely (outside of an office) at least 50% of the time?
#     tech_company: Is your employer primarily a tech company/organization?
#     benefits: Does your employer provide mental health benefits?
#     care_options: Do you know the options for mental health care your employer provides?
#     wellness_program: Has your employer ever discussed mental health as part of an employee wellness program?
#     seek_help: Does your employer provide resources to learn more about mental health issues and how to seek help?
#     anonymity: Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources?
#     leave: How easy is it for you to take medical leave for a mental health condition?
#     mental_health_consequence: Do you think that discussing a mental health issue with your employer would have negative consequences?
#     phys_health_consequence: Do you think that discussing a physical health issue with your employer would have negative consequences?
#     coworkers: Would you be willing to discuss a mental health issue with your coworkers?
#     supervisor: Would you be willing to discuss a mental health issue with your direct supervisor(s)?
#     mental_health_interview: Would you bring up a mental health issue with a potential employer in an interview?
#     phys_health_interview: Would you bring up a physical health issue with a potential employer in an interview?
#     mental_vs_physical: Do you feel that your employer takes mental health as seriously as physical health?
#     obs_consequence: Have you heard of or observed negative consequences for coworkers with mental health conditions in your workplace?
#     comments: Any additional notes or comments
#     var_types = ['continuous',
#                 'nominal',
#                 'continuous',
#                 'continuous',
#                 'continuous',
#                 'continuous',
#                 'continuous',
#                 'continuous',
#                 'continuous',
#                 'continuous',
#                 'continuous',
#                 'continuous',
#                 'continuous',
#                 'continuous',
#                 'continuous',
#                 'continuous',
#                 'continuous',
#                 'continuous',
#                 'continuous',
#                 'continuous',
#                 'continuous',
#                 'continuous',
#                 'nominal',
#                 'nominal'],


mhtech14_common_args = {
    "class_col": "treatment",
    "positive_class": "Yes",
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
}

mhtech14_pd = create_data_provider(
    name="mhtech14",
    file_name="mhtech14.csv.gz",
    sample_size=1.0,
    data_framework="pandas",
    schema=None,
    **mhtech14_common_args,
)

mhtech14_pd.features.drop(columns=["comments"], axis=1, inplace=True)  # type: ignore

mhtech14_pl = create_data_provider(
    name="mhtech14",
    file_name="mhtech14.csv.gz",
    sample_size=1.0,
    data_framework="polars",
    schema=None,
    **mhtech14_common_args,
)

mhtech14_pl.features = mhtech14_pl.features.drop(["comments"])
