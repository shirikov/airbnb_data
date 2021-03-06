# Analysis of Airbnb data

This repo contains the scripts for a short analysis of Airbnb data for selected U.S. cities. The analysis examines what variables can predict review scores and the sentiment of reviews for listed properties. The analysis is conducted as part of the Udacity Data Science Nanodegree course.

## Required Software

Python 3 (e.g., an Anaconda installation) with the following libraries installed: pandas, numpy, sklearn, ast, nltk, requests, bs4, matplotlib, datetime, collections.

## Questions Explored in the Project

This project examines the Airbnb data on properties and user reviews from 2020 to understand:

1. What factors predict more positive review scores for Airbnb properties? Are scores predicted better by host characteristics or by the features of listings themselves?
2. What factors predict more positive review sentiment for Airbnb properties? Is sentiment predicted better by host characteristics or by features of listings themselves?
3. Are the factors that predict review scores and sentiment consistent across different cities?
4. Are the factors that predict review scores and sentiment consistent across different months (times of the year)?

## Files

There are two ways to run the analysis and produce the results:

1. Run everything at once via the *airbnb_analysis_all.ipynb* Jupyter notebook that includes data preparation and the main analysis. The Markdown cells in the notebook provide short explanations for each step. The data preparation step can take a while (10-30 minutes on an average laptop).

2. If you're only interested in looking at the results, you can run the *airbnb_analysis.ipynb* notebook that does not include data preparation and takes only a few seconds to run. The *airbnb_cleaned.csv.gz* CSV file needs to be in the same directory. Alternatively, you can first run the *airbnb_data.py* script (also included) that produces the CSV file with cleaned-up data, and then play around with the *airbnb_analysis.ipynb* notebook.

## Analysis Results

The findings are briefly described in the notebooks and in a blog post [here](https://medium.com/@antonshirikov/who-gets-better-airbnb-reviews-b241f5e04563).

## Acknowledgments

The data used in the analysis are available from Airbnb under Creative Commons; see [here](http://insideairbnb.com/get-the-data.html) for the data and licensing details. The code provided in this repo can be used and modified as needed.