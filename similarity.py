import pandas as pd
import numpy as np
import re
from numpy.linalg import norm
from tqdm import tqdm
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
stoplist = set(stopwords.words('english'))

def get_cosine_similarity(A, B):
    A = np.array(A)
    B = np.array(B)
    cosine_sim = np.dot(A,B)/(norm(A)*norm(B))
    return cosine_sim

def tf_idf_preprocess(text):

    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text)  # Removes numbers and special characters
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Replaces multiple spaces with a single space
    cleaned_text = re.sub(r'\b\w{1}\b', '', cleaned_text)  # Remove single letter words
    cleaned_text = ' '.join([word for word in cleaned_text.split() if word not in stoplist])
    cleaned_text = re.sub('See full summary', '', cleaned_text)
    cleaned_text = cleaned_text.lower()
    return cleaned_text.strip()  # Remove leading and trailing spaces

def create_tfidf_features(corpus, max_features=5000, max_df=0.95, min_df=2):
    tfidf_vectorizor = TfidfVectorizer(decode_error='replace', strip_accents='unicode', analyzer='word',
                                       ngram_range=(1, 1), max_features=max_features,
                                       norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=True,
                                       max_df=max_df, min_df=min_df)
    X = tfidf_vectorizor.fit_transform(corpus)
    print('tfidf matrix successfully created.')
    return X, tfidf_vectorizor

def run_tf_idf_similarity(input_query, data, n_results):
    data['tfidf_description'] = data['tfidf_description'].astype(str)
    data['tfidf_description'] = data['tfidf_description'].apply(
        tf_idf_preprocess)
    n_results = min(n_results, data['tfidf_description'].nunique())
    data = data.drop_duplicates(subset=['tfidf_description']).reset_index(drop = True)
    corpus = data['tfidf_description'].to_list()

    X,v = create_tfidf_features(corpus)

    query = tf_idf_preprocess(input_query)
    query_vec = v.transform([query])
    cosine_similarities = cosine_similarity(query_vec, X)
    cosine_similarities_sorted_index = np.argsort(cosine_similarities, axis=1)
    top_similar_indices = cosine_similarities_sorted_index[::, :-n_results-1:-1]

    final_df = pd.DataFrame({
                'input_query': [input_query] * n_results,
                'tfidf_query': [query] * n_results,
                'original_movie_description':data.loc[top_similar_indices[0]]['description'].to_list(),
                'featured_movie_description': data.loc[top_similar_indices[0]]['tfidf_description'].to_list(),
                'cosine_sim_score' : cosine_similarities[0][top_similar_indices][0]
    })
    return final_df