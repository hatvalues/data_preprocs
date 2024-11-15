# Preprocessed Datasets

A library containing a set of preprocessed datasets, with some clean up and smaller samples of well known datasets from UCI Machine Learning Library, and a few additional items.

The chosen datasets represent a variety of cases suitable for classification problems.

## TODO

It looks like some nulls are in the target variable for YPS. Just need to open up the csv file and drip these rows and save.

## Usage

At the moment, the best way is to clone the repo and e.g. if using Poetry, add `
data-preprocs = {path = "../data_preprocs"}` to the dependences in pyproject.toml

Just import the desired object from the data_providers module.
Each data provider contains a features data frame and a targets series, plus some meta data items, including:

* name: str of the dataset
* sample_size: float no replacement sampling from the original data set
* spiel: str multi-line text with the data set description
* class_col: str the name of the target series
* positive_class: the class value usually associated with a positive outcome (in binary classification)

You can create your own data_provider instance by storing a gzip compress csv file in the default location.
It should have column headers but not row ids.

## Catalogue

* adult
* adult_samp (0.25 sample)
* adult_small_samp (0.025 sample)
* bankmark
* bankmark_samp (0.05 sample)
* breast
* car
* cardio
* cervical (in cervicalh, cervicals, cervicalc and cervicalb variants, representing each of the four different cancer markers as target, excluding the three others)
* cervicalr (a distinct dataset for cerivcal cancer diagnosis)
* credit
* diaretino (diabetes retinopathy)
* german
* heart
* lending_samp (0.1 sample)
* lending_small_samp (0.01 sample)
* lending_tiny_samp (0.0025 sample)
* mhtech14 (2014 Mental Heath in Tech Jobs survey)
* mh1tech16 (2016 Mental Heath in Tech Jobs survey, completely different set of questions). Three questions flag whether the respondent had professional treatment for a mental health issue, and are isolated for three different versions of this data set.
* mh2tech16
* mh3tech16
* mush (mushroom toxicity indentification)
* noshow (medical appointment non-attendance)
* noshow_samp (0.2 sample)
* noshow_small_samp (0.02 sample)
* nursery (multinomial response for nursery place offer)
* nursery_samp (0.2 sample)
* rcdv criminal recidivism
* rcdv_samp (0.1)
* readmit (hospital readmission)
* thyroid (thyroid abnormality)
* thyroid_samp (0.1 sample)
* usoc (understanding society, mental health issues)
* usoc_samp (0.1)
* ypsalc (young people survey, frequency of alcohol consumption)
* ypssmk (frequency of smoking)
