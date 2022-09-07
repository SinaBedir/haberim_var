###############################################################################################
# Libraries
###############################################################################################

from warnings import filterwarnings
from nltk.corpus import stopwords

import pandas as pd
import numpy as np
from urllib.request import urlopen
import json
import warnings
import spacy

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 300)

warnings.simplefilter(action='ignore', category=Warning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

###############################################################################################
# Model Pipeline
###############################################################################################

def data_extractor_from_api(url, excel=False, printer=False):
    response = urlopen(url)
    data_json = json.loads(response.read())
    data = data_json["articles"]
    df = pd.json_normalize(data)

    if excel:
        df.to_excel("nlp_data.xlsx")

    if printer:
        print(data_json)

    return df

def getting_data_pipeline():
    url = "https://newsapi.org/v2/everything?q=apple&from=2022-09-06&to=2022-09-06&sortBy=popularity&apiKey=a298633332b04ca49fbe8450dd6fe353"
    apple_df = data_extractor_from_api(url)

    url = "https://newsapi.org/v2/top-headlines?country=us&category=business&apiKey=a298633332b04ca49fbe8450dd6fe353"
    buss_df = data_extractor_from_api(url)

    url = "https://newsapi.org/v2/top-headlines?sources=techcrunch&apiKey=a298633332b04ca49fbe8450dd6fe353"
    tech_df = data_extractor_from_api(url)

    url = "https://newsapi.org/v2/everything?domains=wsj.com&apiKey=a298633332b04ca49fbe8450dd6fe353"
    wall_df = data_extractor_from_api(url)

    df_list = [apple_df, buss_df, tech_df, wall_df]
    df = pd.concat(df_list, axis=0)
    df = df.reset_index()
    df.drop("index", axis=1, inplace=True)

    return df

def text_preprocessing(df):
    df['content'] = df['content'].str.lower()
    df['content'] = df['content'].str.replace('[^\w\s]', '')
    df['content'] = df['content'].str.replace('\d', '')
    sw = stopwords.words('english')
    df['content'] = df['content'].apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))
    temp_df = pd.Series(' '.join(df['content']).split()).value_counts()
    drops = temp_df[temp_df <= 1]
    df['content'] = df['content'].apply(lambda x: " ".join(x for x in x.split() if x not in drops))
    return df

def model_pipeline():
    df = getting_data_pipeline()
    df = text_preprocessing(df)
    nlp = spacy.load('en_core_web_sm')
    nlp_docs = []

    for nlp_doc in df["content"]:
        nlp_docs.append(nlp(nlp_doc))

    col1 = []
    col2 = []
    col3 = []
    col4 = []
    col3_title = []
    col4_title = []

    for i in nlp_docs:
        for y in nlp_docs:
            col1.append(i)
            col2.append(y)

    df = getting_data_pipeline()

    for i in df["content"]:
        for y in df["content"]:
            col3.append(i)
            col4.append(y)

    for i in df["title"]:
        for y in df["title"]:
            col3_title.append(i)
            col4_title.append(y)

    nlp_df = pd.DataFrame({"col1": col1, "col2": col2, "col3": col3,
                           "col4": col4, "col3_title": col3_title, "col4_title": col4_title})

    drop_rows = nlp_df[nlp_df["col3_title"] == nlp_df["col4_title"]]
    nlp_df.drop(drop_rows.index, axis=0, inplace=True)
    nlp_df = nlp_df.reset_index()
    nlp_df.drop("index", axis=1, inplace=True)

    scores = []
    for i in range(0, len(nlp_df)):
        scores.append(nlp_df["col1"][i].similarity(nlp_df["col2"][i]))

    scores = pd.DataFrame(scores, columns=["Similarity_Scores"])
    nlp_df = pd.concat([nlp_df, scores], axis=1)

    nlp_df.drop_duplicates(inplace=True)

    return nlp_df

def news_recommender(new_title, rec_count=10):
    nlp_df = model_pipeline()
    rec_df = nlp_df[nlp_df['col3_title'].str.contains(new_title) == True].sort_values(by="Similarity_Scores", ascending=False)[
             0:rec_count]
    print(rec_df["col4_title"])

new_title = "Samsung's 32-inch Smart Monitor M8 falls to a new low"
news_recommender(new_title)