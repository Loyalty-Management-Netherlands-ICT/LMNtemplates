"""
Python Functions for LMN

Joël Gastelaars
June 2021 | last update: April 2022
"""

# Function to automatically install packages when not available
import pip
def import_or_install(package):
    try:
        __import__(package)
        print("imported")
    except ImportError:
        pip.main(['install', '--user', package])
        print("installed")

# Installing basic packages for the first time
import_or_install('openpyxl')
import_or_install('re')
import_or_install('pandas')
import_or_install('numpy')
import_or_install('seaborn')
import_or_install('plotly')
import_or_install('bokeh')
import_or_install('matplotlib')
import_or_install('scikit-learn')
import_or_install('nltk')
import_or_install('gensim')
import_or_install('keras')
import_or_install('tensorflow')
import_or_install('streamlit')
import_or_install('hydralit')
import_or_install('dotenv')
import_or_install('teradatasql')
import_or_install('dotenv')
import_or_install('teradataml')
import_or_install('streamlit_bokeh_events')

import os
import time
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import teradatasql
# When importing hydralit you will run into an ModuleNotFoundError: No module named 'streamlit.script_run_context.
# Resolve this error by going into the hydralit code to the sessionstate.py file and adjust codeline 8 as follows:
# R -> File --> open file
# Klik op drie puntjes rechts (naast R logo)
# Kopieer path naar hydralit library. Bij mij is dit: ~/.local/lib/python3.8/site-packages/hydralit
# Open sessionstate.py file, adapt below and save
# change: from streamlit.script_run_context import get_script_run_ctx
# To: from streamlit.scriptrunner import get_script_run_ctx
# see, for more info; https://github.com/TangleSpace/hydralit/issues/27
# NOTE THAT this issue may already be resolved in the future through an open Pull Request on Github
import hydralit as hy
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dotenv import load_dotenv
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem.snowball import SnowballStemmer #better than Porterstemmer, in all languages 
import gensim
from gensim.models import CoherenceModel
from gensim.utils import simple_preprocess
from pandas_profiling import ProfileReport
# ---------------------------------------------------------------------------
# STANDARD FUNCTIONS
# ---------------------------------------------------------------------------

###
# DATABASE CONNECTION IN PYTHON
###

###
#   Step 1:
#       1) In VSC go to your root directory (in my case that is /home/bram_mosterd)
#       2) Create a new file in there named: '.env' (yes, no name just .env)
#       3) Copy and paste your credentials for db (or simply paste the content of your .Renviron into the .env file )
#       4) Save the file (ctrl + s)
###

###
#   Step 2: Test Connection
###
def test_db_connection():
    # load .env file (make sure load_dotenv() module and teradatasql are imported)
    _ = load_dotenv()
    db_host = os.getenv('td_prod1')
    db_username = os.getenv('td_user')
    db_password = os.getenv('td_prod_pw')
    
    # Testing query this can of course be anything else.
    query = """SELECT TOP 50 * FROM ATP_DBM_VIEW.transactie 
                WHERE transactiedatum > current_date() - 3
                AND partnerid = 400678;"""
    try:
        print("Testing whether db connection and test query works.")
        df = run_query(query, db_host, db_username, db_password)
        print(f"Successfully made a connection to db with credentials. As a test the following data was successfully loaded: \n{df.head()}")
    except Exception as e:
        print(f"Someething went wrong when trying to connect to the database: {e}")


def run_query(query, db_host, db_username, db_password):
    """
        Function that runs input SQL query and loads it into a dataframe
    """
    with teradatasql.connect(host=db_host, user=db_username, password=db_password) as connect:  
        data = pd.read_sql(query, connect)
    return data


def get_transaction_data(host, user, passw):
    """
        Aggregates all Air Miles issued / redemption from transactie data per month, partner and transactiereden.
        Returns sorterd dataframe
    """
  query = """SELECT partnernaam, transactiedatum_isojaar, transactiedatum_isoweek, partner_bpcode, partner_aggregatie, partner_aggregatie2, transactiereden, sum(punten) FROM ATP_DBM_VIEW.transactie
    WHERE transactiedatum_isojaar IN (2021, 2022)
    GROUP BY 1, 2, 3, 4, 5, 6, 7
    ORDER BY 1, 2, 3, 4, 5, 6, 7;"""
  # use with to immediately close connection to db
  with teradatasql.connect(host=host, user=user, password=passw) as connect:
     df = pd.read_sql(query, connect)
  df = df.sort_values(['transactiedatum_isojaar', 'transactiedatum_isoweek', 'partnernaam'])
  return df


def correlation_heatmap(df: pd.DataFrame, cols: list, save_to_fig: bool):
    """
        Calculates correlations for cols in df and returns heatmap with correlations.
        Set save_to_fig = True if you want to save the figure.

        More useful info: https://medium.com/@szabo.bibor/how-to-create-a-seaborn-correlation-heatmap-in-python-834c0686b88e
    """
    plt.figure(figsize=(16, 6))
    plot = sns.heatmap(df[cols].corr(),
                        vmin=-1,
                        vmax=1,
                        annot=True,
                        cmap='BrBG' )
    
    if save_to_fig:
        file_name = "correlation_heatmap.png"
        plt.savefig(file_name)
        print(f"Saved to: {os.getcwd() + '/' + file_name}")
    return plot

#BM: Add underscore as well to be replaced? 
def remove_trail_patterns(text):
    text = str(text).replace("\t"," ").replace("\n"," ").replace("-"," ").strip()
    return text

def tolower(text):
    text = str(text).lower()
    return text

def remove_some_punc(text):
    text = str(text).replace("=", " ").replace(":", " ").strip()
    return text

def remove_lead_zeros(text):
    text = str(text).lstrip("([0]+)\w+")
    return text

def remove_lead_chr(text):
    text = str(text).lstrip("(^[b]['])")
    return text

def remove_punctuation(text):
    # If not matched with a word (\w) or whitespace character (\s) set to empty.
    text = re.sub(r"[^\w\s]","", text)
    return text

def remove_digits(text):
    text = re.sub(r"[^A-Za-z\s]", "", text)
    return text

def remove_single_char(text):
    if type(text) == str:
        text = " ".join([w for w in text.split() if len(w)>1] )
    return text

def preprocessing(text):
    if type(text) == str:
        text = text.lower()
        text = remove_punctuation(text)
        text = remove_digits(text)
        text = remove_single_char(text)
        # Currently decided not to use stemming due to understandability of final keywords
        # TODO: find a way to get back to understandable context with stemmed words
        # text = tokenize_and_stem(text) 
        return text
    else:
        return ""

def tokenize_and_stem(text):
    stemmer = SnowballStemmer("english") # is better than PorterStemmer
    # stemmer = PorterStemmer()
    stemmed = []
    if isinstance(text,str):
        text = nltk.word_tokenize(text)
        for word in text:
            stemmed.append(stemmer.stem(word))
        return stemmed
    else:
        return [""]
      
# What's the purpose of this function? Concatenate all elements in list to str?
# if so: "".join(str(x) for x in text) may be better
def list_to_str(text):
    text = text.astype(str).str.replace("[","").str.replace("]","").str.replace("\"","").str.replace(",","") 
    return text

def set_dt_aware(text):
    text = pd.to_datetime(text, utc = True, errors = "coerce")
    return text

def set_tz_naive(timestamp):
    timestamp = timestamp.apply(lambda x: x.tz_localize(None))
    return timestamp
    
#### TF-IDF and Topic Modelling functions ####
def get_tfidf(docs, stopwords): 
    start = time.perf_counter()
     # ngram_range (1,3) takes unigrams, bigrams and trigrams 
     # lowercase false, because list and already pre-processed anyway
    tf = TfidfVectorizer(lowercase=False, analyzer="word", ngram_range=(1,3), min_df=2, max_df=.9, stop_words = stopwords)
    tfidf = tf.fit_transform(docs)
    feature_names = tf.get_feature_names() 
    #df = pd.DataFrame(tfidf.todense().tolist(), columns=feature_names)
    print ("\nGot TF-IDF matrix in {} seconds.".format(round(time.perf_counter()-start,4)))
    return tfidf, tf, feature_names

#http://na-o-ys.github.io/others/2015-11-07-sparse-vector-similarities.html
def get_cosmat(mat):
    start = time.perf_counter()
    cosmat = cosine_similarity(mat)
    print("Got cosinematrix in {} seconds.".format(round(time.perf_counter()-start,4))) 
    return cosmat

#https://bergvca.github.io/2017/10/14/super-fast-string-matching.html
def get_matches_df(sparse_matrix, similaritythreshold):
    start = time.perf_counter()
    matches = [(i, j, sparse_matrix[i,j]) for i, j in zip(*sparse_matrix.nonzero())]

    result = pd.DataFrame(matches, columns = ["action","related_action","similarity"])
    # remove all with similarity < threshold
    result = result[result["similarity"] > similaritythreshold]
    # remove all with action == related action
    result = result.query("action != related_action")
    result = (result.
                sort_values("similarity", ascending=False).
                reset_index().
                drop("index",1))
    print("Got results in {} seconds.".format(round(time.perf_counter()-start,4))) 
    return result

def plot_10_most_common_words(count_data, tf_vectorizer):
    words = tf_vectorizer.get_feature_names()
    total_counts = np.zeros(len(words))
    for t in count_data:
        total_counts+=t.toarray()[0]
    
    count_dict = (zip(words, total_counts))
    count_dict = sorted(count_dict, key=lambda x:x[1], reverse=True)[0:10]
    words = [w[0] for w in count_dict]
    counts = [w[1] for w in count_dict]
    x_pos = np.arange(len(words)) 
    
    plt.figure(2, figsize=(15, 15/1.6180))
    plt.subplot(title="10 most common words")
    sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
    sns.barplot(x_pos, counts, palette="husl")
    plt.xticks(x_pos, words, rotation=90) 
    plt.xlabel("words")
    plt.ylabel("counts")
    plt.show()

# Function to display topics for NMF & LDA
def print_topics(model, tf_vectorizer, n_top_words):
    words = tf_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #%d:" % topic_idx)
        print(" ".join([words[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))

# Gensim topic modelling functions
def sent_to_words(sentences): # Tokenization and more cleaning
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

def remove_stopwords(texts, stopwords):
    return [[word for word in simple_preprocess(str(doc)) if word not in stopwords] for doc in texts]

def make_bigrams(bigram, texts):
    return [bigram[doc] for doc in texts]

def make_trigrams(trigram, bigram, texts):
    return [trigram[bigram[doc]] for doc in texts]

def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=dictionary,
                                           num_topics=num_topics,
                                           random_state=100,
                                           update_every=0, # full batch learning
                                           #chunksize=1000, # only needed if online learning
                                           passes=10,
                                           alpha="auto",
                                           per_word_topics=False)
        model_list.append(model)
        coherencemodel = CoherenceModel(model = model, texts = texts, dictionary = dictionary, coherence = "c_v")
        coherence_values.append(coherencemodel.get_coherence())
    return model_list, coherence_values

# Get the dominant topic, % and keywords per row (response)
def format_topics_sentences(ldamodel, corpus, texts):
    sent_topics_df = pd.DataFrame()
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ["Dominant_Topic", "Perc_Contribution", "Topic_Keywords"]

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)
