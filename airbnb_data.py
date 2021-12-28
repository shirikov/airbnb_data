import pandas as pd
import numpy as np
import ast
import re
from datetime import datetime
from collections import Counter
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import requests
from bs4 import BeautifulSoup

# Download word scores for the sentiment analysis part
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()
imp = IterativeImputer(max_iter=10, random_state=42)

# Get the data
# Scrape URLs to specific Airbnb data sets
reqs = requests.get('http://insideairbnb.com/get-the-data.html')
soup = BeautifulSoup(reqs.text, 'html.parser')
 
data_urls = [link.get('href') for link in soup.find_all('a')]
data_urls = [link for link in data_urls if link is not None]

# Select particular cities
data_urls = [link for link in data_urls if ('tx/austin' in link or 
                                            'ma/boston' in link or
                                            'il/chicago' in link or
                                            'or/portland' in link)]

def check_month(url_full, month_list):
    
    '''Check whether url contains a particular month.'''
    
    date_compiled = re.search(r'\d{4}-\d{2}-\d{2}', url_full).group()
    url_month = datetime.strptime(date_compiled, '%Y-%m-%d').month
    return(url_month in month_list)

# Keep records for January, April, July, and October
data_urls = [link for link in data_urls if check_month(link,
                                                       [1, 4, 7, 10])]

# Split lists of URLs into listings and reviews data
data_urls_prop = [link for link in data_urls if 'listings.csv.gz' in link]
data_urls_rev = [link for link in data_urls if 'reviews.csv.gz' in link]


# Get data for specific cities and dates
def read_airbnb_data(data_url):
    
    '''Read in data from Airbnb for selected cities and dates.'''
    
    adf = pd.read_csv(data_url, index_col=None, header=0)
    
    # Drop variables unlikely to be useful in prediction
    # or with a lot of missing values that are difficult to impute
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

# Recode/transform variables

# Create dummies for amenities mentioned in listings
# Count the frequency for each amenity
amenities_list = []
for i in range(0, prop_df.shape[0]):
    amenities_list.extend(ast.literal_eval(prop_df.amenities[i]))
amenities_count = Counter(amenities_list)

# Keep amenities that are not mentioned too rarely or too frequently
amenities_frequent = dict()
for (key, value) in amenities_count.items():
   if value > 10000 and value < 80000 and not ('TV' in key):
       amenities_frequent[key] = value

for amenity in list(amenities_frequent.keys()):
    amenity_name = "am_" + amenity.lower()
    amenity_name = amenity_name.replace(" ", "_").replace("-", "_")
    prop_df[amenity_name] = prop_df.amenities.str.contains(amenity).astype(int)

# Convert variables into dummies
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

# Add dummy for whether host neighborhood is indicated
prop_df['host_neigh_present'] = prop_df[
    'host_neighbourhood'
    ].isnull().astype(int)

# Recode price as numeric
prop_df['price'] = prop_df['price'].str.replace("\$", "")
prop_df['price'] = pd.to_numeric(prop_df['price'], errors='coerce')

# Recode host_since into days
prop_df['host_since_days'] = pd.to_datetime(prop_df['host_since'])
prop_df['host_since_days'] = (pd.to_datetime('2021-10-20') 
                              - prop_df['host_since_days']).dt.days

# Add property, neighborhood, and host description lengths
prop_df['desc_length'] = prop_df['description'].str.len().fillna(0)
prop_df['neigh_desc_length'] = prop_df[
    'neighborhood_overview'
    ].str.len().fillna(0)
prop_df['host_desc_length'] = prop_df['host_about'].str.len().fillna(0)

# Deal with missing data 

# Check missing values by column
prop_df.isnull().sum()

# Replace nans by zeros for bedrooms 
prop_df['bedrooms'] = prop_df['bedrooms'].fillna(0)

# Impute beds and price (strongly correlated)
imp.fit(prop_df[['beds', 'bedrooms', 'price']])
prop_df_imp = imp.transform(prop_df[['beds', 'bedrooms', 'price']])
prop_df_imp = pd.DataFrame(prop_df_imp, columns = ['beds', 'bedrooms', 
                                                   'price'])

# Round imputed beds and bedrooms
prop_df_imp[['beds', 'bedrooms']] = prop_df_imp[['beds', 'bedrooms']].round(0)
prop_df[['beds', 'bedrooms', 'price']] = prop_df_imp[['beds', 'bedrooms', 
                                                      'price']]

# Drop rows with missing values
prop_df = prop_df.dropna(subset=['review_scores_rating', 'rev_sent',
                                 'host_name', 'host_since', 
                                 'host_total_listings_count',
                                 'host_identity_verified'])
prop_df = prop_df[prop_df.price != 0].reset_index(drop=True)

# Clean up column names
prop_df.columns = prop_df.columns.str.lower()
prop_df.columns = prop_df.columns.str.replace(' ', '_')
prop_df.columns = prop_df.columns.str.replace('\/', "_")

# Put review scores on the same scale 
# (they are on different scales in different months)
prop_df['review_scores_rating'] = np.where((prop_df['month_january'] == 1) | 
                                           (prop_df['month_april'] == 1),
                                           prop_df['review_scores_rating'] / 20, 
                                           prop_df['review_scores_rating'])

# Save the cleaned-up data set to csv
prop_df.drop(columns=[
    'description', 'neighborhood_overview', 'host_about', 
    'host_neighbourhood', 'amenities'
    ]).to_csv('airbnb_cleaned.csv.gz',
              index=False, 
           compression="gzip")