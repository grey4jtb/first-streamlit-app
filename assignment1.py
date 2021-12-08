"""


@author: hamzafarooq@ MABA CLASS
"""

import streamlit as st
import pandas as pd

import plotly.express as px
#from spacy.lang.en.product_name import product_name
import pickle as pkl
from tqdm import tqdm
import re
from summarizer import Summarizer

# Define Constants
HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""
#prod_aisles = pd.read_csv('aisles.csv.zip', header=0)

prod_product = pd.read_csv('"C:\Users\16128\Documents\GitHub\first-streamlit-app\products.csv"products.csv', header=0)
# Define functions
def lower_case(input_str):
    input_str = input_str.lower()
    return input_str

# Define Summarizer model for provide review summary
#model = Summarizer()

#def summarized_review(data):
    #data = data.values[0]
    #return model(data, num_sentences=3)


class HotelRecs:

    def __init__(self):
        # Define embedder
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

    def clean_data(self):
        # Aggregate all reviews for each hotel
        df_agg_reviews = prod_product.sort_values(['product_name']).groupby('department_id', sort=False).review_body.apply(
            ''.join).reset_index(name='review_body')

        # Generate review summary
        df_agg_summary = df_agg_reviews.copy()
        #df_agg_summary['summary'] = df_agg_summary[["review_body"]].apply(summarized_review, axis=1)

# Retain only alpha numeric characters
        df_agg_reviews['review_body'] = df_agg_reviews['review_body'].apply(lambda x: re.sub('[^a-zA-z0-9\s]', '', x))

        # Change to lowercase
        df_agg_reviews['review_body'] = df_agg_reviews['review_body'].apply(lambda x: lower_case(x))

        # Remove stop words
        #df_agg_reviews['review_body'] = df_agg_reviews['review_body'].apply(lambda x: ' '.join([item for item in x.split() if item not in stopwords]))

        # Retain the parsed review body in the summary df
        df_agg_summary['review_body'] = df_agg_reviews['review_body']

        df_sentences = df_agg_reviews.set_index("review_body")
        df_sentences = df_sentences["hotelName"].to_dict()
        df_sentences_list = list(df_sentences.keys())

        # Embeddings
        corpus = [str(d) for d in tqdm(df_sentences_list)]
        corpus_embeddings = self.embedder.encode(corpus, convert_to_tensor=True)

        # Dump to pickle file to use later for prediction
        with open("corpus.pkl", "wb") as file1:
            pkl.dump(corpus, file1)

        with open("corpus_embeddings.pkl", "wb") as file2:
            pkl.dump(corpus_embeddings, file2)

        with open("df_agg_reviews.pkl", "wb") as file3:
            pkl.dump(df_agg_reviews, file3)

        with open("df_agg_summary.pkl", "wb") as file4:
            pkl.dump(df_agg_summary, file4)

        return df_agg_summary, df_agg_reviews, corpus, corpus_embeddings

    def construct_app(self):
        df_agg_summary, df_agg_reviews, corpus, corpus_embeddings = self.clean_data()

st.title("Final Project Presentation")
st.markdown("Joseph Boykin")
st.markdown(" ")
st.title(" The Application: ITEM LOCATOR ")
st.markdown(" ")
st.markdown("My app is designed to help consumers navigate their way around retail stores to locate their specific items to purchase.  This will save consumers time wondering around searching for item.")
st.markdown("● The customer enters the items they are searching for.")
st.markdown("● The app will locate the items in the store and let the customer know where the items are located.")
st.markdown(" ")
st.title("Final Project Reflection")
st.markdown(" ")
st.markdown("● Appreciation to Santoshi for assisting me with my final project!")
st.markdown("● Need to better advocate for myself. ")
