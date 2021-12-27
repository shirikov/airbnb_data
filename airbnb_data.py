import pandas as pd
import numpy as np
import ast
import re
from datetime import datetime
from collections import Counter
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from nltk.sentiment import SentimentIntensityAnalyzer
import requests
from bs4 import BeautifulSoup
 
#import nltk
#nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()
imp = IterativeImputer(max_iter=10, random_state=42)

import os
os.chdir('Box Sync')
os.chdir('Udacity DS Nanodegree')
os.chdir('Airbnb')

# Get the data
# Scrape links to Airbnb data sets
reqs = requests.get('http://insideairbnb.com/get-the-data.html')
soup = BeautifulSoup(reqs.text, 'html.parser')
 
data_urls = [link.get('href') for link in soup.find_all('a')]
data_urls = [link for link in data_urls if link is not None]

# Select particular cities and dates
data_urls = [link for link in data_urls if ('tx/austin' in link or 
                                            'ma/boston' in link or
                                            'il/chicago' in link or
                                            'or/portland' in link)]

def check_month(url_full, month_list):
    
    '''Check whether url contains a particular month.'''
    
    date_compiled = re.search(r'\d{4}-\d{2}-\d{2}', url_full).group()
    url_month = datetime.strptime(date_compiled, '%Y-%m-%d').month
    return(url_month in month_list)
    
data_urls = [link for link in data_urls if check_month(link,
                                                       [1, 4, 7, 10])]

# Separate out into listings and reviews data
data_urls_prop = [link for link in data_urls if 'listings.csv.gz' in link]
data_urls_rev = [link for link in data_urls if 'reviews.csv.gz' in link]


# Get data for specific cities and dates
def read_airbnb_data(data_url):
    
    '''Read in data from Airbnb for selected cities and dates.'''
    
    adf = pd.read_csv(data_url, index_col=None, header=0)
    
    if 'listings.csv.gz' in data_url:
        adf = adf.drop(columns = [
            'listing_url', 'scrape_id', 'last_scraped', 'picture_url',
            'host_url', 'host_location', 'host_thumbnail_url', 
            'host_picture_url', 'neighbourhood', 'host_acceptance_rate',
            'host_response_rate', 'host_response_time',
            'neighbourhood_group_cleansed', 'latitude', 'longitude',
            'bathrooms', 'bathrooms_text', 'accommodates', 'property_type',
            'minimum_nights', 'maximum_nights', 'minimum_minimum_nights',
            'maximum_minimum_nights', 'minimum_maximum_nights',
            'maximum_maximum_nights', 'minimum_nights_avg_ntm',
            'maximum_nights_avg_ntm', 'calendar_updated', 
            'has_availability', 'availability_30', 'availability_60',
            'availability_90', 'availability_365', 'calendar_last_scraped',
            'license', 'host_verifications', 
            'host_has_profile_pic', 'first_review', 'last_review',
            'host_listings_count', 'calculated_host_listings_count',
            'calculated_host_listings_count_entire_homes',
            'calculated_host_listings_count_private_rooms',
            'calculated_host_listings_count_shared_rooms',
            'reviews_per_month', 'number_of_reviews',
            'number_of_reviews_ltm', 'number_of_reviews_l30d',
            'review_scores_accuracy', 'review_scores_cleanliness',
            'review_scores_checkin', 'review_scores_communication',
            'review_scores_location', 'review_scores_value'
            ], axis=1)
    
    if 'reviews.csv.gz' in data_url:
        adf = adf.drop(columns=['id', 'date', 'reviewer_id', 'reviewer_name'], 
                       axis=1)
        
    data_city = re.split(r'\d{4}-\d{2}-\d{2}', data_url)[0]
    data_city = re.split('/(.*?)/', data_city)[-2].capitalize()
    
    data_date = re.search(r'\d{4}-\d{2}-\d{2}', data_url).group()
    
    adf['City'] = data_city
    adf['Date'] = data_date
    adf['Month'] = datetime.strptime(data_date, '%Y-%m-%d').strftime("%B")

    return(adf)

# Read data on listings (properties) and combine in a single data set
prop_df = [read_airbnb_data(link) for link in data_urls_prop]
prop_df = pd.concat(prop_df, axis=0, ignore_index=True)

# Read reviews data and combine in a single data set
reviews_df = [read_airbnb_data(link) for link in data_urls_rev]
reviews_df = pd.concat(reviews_df, axis=0, ignore_index=True)

# Transform and clean the data
# Calculate sentiment (polarity score) for each review
# https://realpython.com/python-nltk-sentiment-analysis/
def get_polarity_score(review_text):
    
    '''Returns polarity scores for words in Airbnb comments.'''
    
    try:
        review_polarity = sia.polarity_scores(review_text)
        return review_polarity['compound']
    except:
        return np.nan
   
reviews_df['rev_sent'] = reviews_df[
    'comments'
    ].apply(get_polarity_score)

# Calculate average sentiment by property
sentiment_by_listing = reviews_df.groupby(
    'listing_id'
    ).rev_sent.mean().reset_index()

# Join reviews data with the data on listings
prop_df = pd.merge(
    prop_df, sentiment_by_listing, left_on='id', right_on='listing_id',
    how='left'
    ).drop(columns=['listing_id'])

# Recode variables
# Create dummies for amenities mentioned in listings
# Count the frequency for each amenity
amenities_list = []
for i in range(0, prop_df.shape[0]):
    amenities_list.extend(ast.literal_eval(prop_df.amenities[i]))
amenities_count = Counter(amenities_list)

# Keep only amenities mentioned in at least 3000 listings
amenities_frequent = dict()
for (key, value) in amenities_count.items():
   if value > 10000 and value < 80000 and not ('TV' in key):
       amenities_frequent[key] = value

for amenity in list(amenities_frequent.keys()):
    amenity_name = "am_" + amenity.lower()
    amenity_name = amenity_name.replace(" ", "_").replace("-", "_")
    prop_df[amenity_name] = prop_df.amenities.str.contains(amenity).astype(int)

# Recode price as numeric
prop_df['price'] = prop_df['price'].str.replace("\$", "")
prop_df['price'] = pd.to_numeric(prop_df['price'], errors='coerce')

# EXPLAIN WHY I'M NOT USING THESE - LOSING A LOT OF DATA, AND UNCLEAR IF THESE ARE RELATED
# UNCLEAR WHY DATA ARE MISSING
# AND RATES ARE HEAVILY CONCENTRATED CLOSE TO 100%; RESPONSE TIME - WITHIN ONE/SEVERAL HOURS
# SO NOT INFORMATIVE ANYWAY
# Recode response and acceptance rate as numeric
# prop_df['host_response_rate'] = prop_df[
#     'host_response_rate'
#     ].str.replace("\%", "")
# prop_df['host_acceptance_rate'] = prop_df[
#     'host_acceptance_rate'
#     ].str.replace("\%", "")
# # Create dummies for missing response/acceptance rate
# prop_df['host_response_rate_missing'] = prop_df[
#     'host_response_rate'
#     ].isnull().astype(int)
# prop_df['host_acceptance_rate_missing'] = prop_df[
#     'host_acceptance_rate'
#     ].isnull().astype(int)
# prop_df['host_response_rate'] = pd.to_numeric(prop_df[
#     'host_response_rate'
#     ], errors='coerce')
# prop_df['host_acceptance_rate'] = pd.to_numeric(prop_df[
#     'host_acceptance_rate'
#     ], errors='coerce')

# Add property, neighborhood, and host description lengths
prop_df['desc_length'] = prop_df['description'].str.len().fillna(0)
prop_df['neigh_desc_length'] = prop_df[
    'neighborhood_overview'
    ].str.len().fillna(0)
prop_df['host_desc_length'] = prop_df['host_about'].str.len().fillna(0)

# Add dummy for whether host neighborhood is indicated
prop_df['host_neigh_present'] = prop_df[
    'host_neighbourhood'
    ].isnull().astype(int)

# Recode host_since into days
prop_df['host_since_days'] = pd.to_datetime(prop_df['host_since'])
prop_df['host_since_days'] = (pd.to_datetime('2021-10-20') 
                              - prop_df['host_since_days']).dt.days

# Convert some variables into dummies
prop_df['host_identity_verified'] = (prop_df[
    'host_identity_verified'
    ] == 't').astype(int)
prop_df['instant_bookable'] = (prop_df[
    'instant_bookable'
    ] == 't').astype(int)
prop_df['host_is_superhost'] = (prop_df[
    'host_is_superhost'
    ] == 't').astype(int)

# Create dummies for room types
prop_df = pd.get_dummies(prop_df, columns=['room_type'], 
                         prefix='room_type')

# Create a dummy that captures whether a host has more than 
# 3 properties listed
prop_df['host_many_listings'] = (prop_df['host_total_listings_count'] 
                                 > 3).astype(int)

# Create dummies for city and month
prop_df = pd.get_dummies(prop_df, columns=['City'], prefix='city')
prop_df = pd.get_dummies(prop_df, columns=['Month'], prefix='month')

# Replace nans by zeros for bedrooms 
prop_df['bedrooms'] = prop_df['bedrooms'].fillna(0)
# impute: beds, bedrooms (well predicted by price), price
# https://machinelearningmastery.com/iterative-imputation-for-missing-values-in-machine-learning/
# https://scikit-learn.org/stable/modules/impute.html
imp.fit(prop_df[['beds', 'bedrooms', 'price']])
prop_df_imp = imp.transform(prop_df[['beds', 'bedrooms', 'price']])
prop_df_imp = pd.DataFrame(prop_df_imp, columns = ['beds', 'bedrooms', 'price'])
beds_missings = prop_df[prop_df['beds'].isnull()].index.tolist()
gg = prop_df_imp.iloc[beds_missings]
bedrooms_missings = prop_df[prop_df['bedrooms'].isnull()].index.tolist()
hh = prop_df_imp.iloc[bedrooms_missings ]
# Round imputed beds and bedrooms
prop_df_imp[['beds', 'bedrooms']] = prop_df_imp[['beds', 'bedrooms']].round(0)
prop_df[['beds', 'bedrooms', 'price']] = prop_df_imp[['beds', 'bedrooms', 'price']]

# Clean up column names
prop_df.columns = prop_df.columns.str.lower()
prop_df.columns = prop_df.columns.str.replace(' ', '_')
prop_df.columns = prop_df.columns.str.replace('\/', "_")

# Drop rows with missing values
prop_df = prop_df.dropna(subset=['review_scores_rating', 'rev_sent',
                                 'host_name', 'host_since', 
                                 'host_total_listings_count',
                                 'host_identity_verified'])
prop_df = prop_df[prop_df.price != 0].reset_index(drop=True)

# Standardize review scores (they are on a different scale in different months)
prop_df['review_scores_rating'] = (
    prop_df['review_scores_rating']/20
    ).where((prop_df.month_january==1) | (prop_df.month_april==1))

prop_df['review_scores_rating'] = np.where((prop_df['month_january'] == 1) | 
                                           (prop_df['month_april'] == 1),
                                           prop_df['review_scores_rating'] / 20, 
                                           prop_df['review_scores_rating'])

# Save to csv
prop_df.drop(columns=[
    'description', 'neighborhood_overview', 'host_about', 
    'host_neighbourhood', 'amenities'
    ]).to_csv('airbnb_cleaned.csv.gz',
              index=False, 
           compression="gzip")