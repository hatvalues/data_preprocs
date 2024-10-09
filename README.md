# Preprocessed Datasets

A library containing a set of preprocessed datasets, with some clean up and smaller samples of well known datasets from UCI Machine Learning Library, and a few additional items.

The chosen datasets represent a variety of cases suitable for classification problems.

## Usage

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
* mhtech16 (2016 Mental Heath in Tech Jobs survey, completely different set of questions)
