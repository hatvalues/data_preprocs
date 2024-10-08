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

pandas_schema = {
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


polars_schema = {
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
    schema=pandas_schema,
    **bankmark_common_args,
)


bankmark_samp_pd = create_data_provider(
    name="bankmark_samp",
    file_name="bankmark_samp.csv.gz",
    sample_size=0.05,
    schema=pandas_schema,
    **bankmark_common_args,
)

bankmark_pl = create_data_provider(
    name="bankmark",
    file_name="bankmark.csv.gz",
    sample_size=1.0,
    schema=polars_schema,
    **bankmark_common_args,
)


bankmark_samp_pl = create_data_provider(
    name="bankmark_samp",
    file_name="bankmark_samp.csv.gz",
    sample_size=0.05,
    schema=polars_schema,
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

cervicalh_pd.features.drop(columns=["Schiller", "Citology", "Biopsy"], axis=1, inplace=True) # type: ignore
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

cervicals_pd.features.drop(columns=["Hinselmann", "Citology", "Biopsy"], axis=1, inplace=True) # type: ignore
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

cervicalc_pd.features.drop(columns=["Hinselmann", "Schiller", "Biopsy"], axis=1, inplace=True) # type: ignore
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

cervicalb_pd.features.drop(columns=["Hinselmann", "Schiller", "Citology"], axis=1, inplace=True) # type: ignore
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
    **cervicalr_common_args, # type: ignore
)

cervicalr_pl = create_data_provider(
    data_framework="polars",
    schema=None,
    **cervicalr_common_args, # type: ignore
)
