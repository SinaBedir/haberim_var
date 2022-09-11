###############################################################################################
# Libraries
###############################################################################################

from warnings import filterwarnings
import warnings

warnings.simplefilter(action='ignore', category=Warning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import numpy as np
from urllib.request import urlopen
import json
import spacy
import mysql.connector
import os
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 300)

###############################################################################################
# Model Pipeline
###############################################################################################
get_messages_sql = """
SELECT * FROM haberimvar.Haber 
WHERE 
YEAR(DATE_) + MONTH(DATE_) + DAY(DATE_) 
=
YEAR(DATE_SUB(CURDATE(), INTERVAL 1 DAY)) +
MONTH(DATE_SUB(CURDATE(), INTERVAL 1 DAY)) +
DAY(DATE_SUB(CURDATE(), INTERVAL 1 DAY))
ORDER BY ID ASC
"""

def connect_rds():
    cnx = mysql.connector.connect(host='final.cluster-ciwzdfrp1kms.eu-central-1.rds.amazonaws.com',
                                  user='awsbc2',
                                  passwd='haydegidelum',
                                  database='haberimvar',
                                  port='63306')
    cur = cnx.cursor()
    cur.execute(get_messages_sql)
    res = cur.fetchall()
    df = pd.DataFrame(res)

    cnx.close()

    return df

df = connect_rds().drop_duplicates(subset=2)
df = df.rename(columns={0: "id",
                       1: "title",
                       2: "content",
                       3: "date"})

sw = stopwords.words('english')

def text_preprocessing(df, sw):
    df['content'] = df['content'].str.lower()
    df['content'] = df['content'].str.replace('[^\w\s]', '')
    df['content'] = df['content'].str.replace('\d', '')
    df['content'] = df['content'].apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))
    temp_df = pd.Series(' '.join(df['content']).split()).value_counts()
    drops = temp_df[temp_df <= 1]
    df['content'] = df['content'].apply(lambda x: " ".join(x for x in x.split() if x not in drops))
    return df

df = text_preprocessing(df, sw)
old_df = connect_rds()
nlp = spacy.load('en_core_web_sm')

def model_pipeline(df, old_df, nlp):

    nlp_docs = []

    for nlp_doc in df["content"]:
        nlp_docs.append(nlp(nlp_doc))

    col1 = []
    col2 = []
    col3 = []
    col4 = []
    col3_title = []
    col4_title = []
    col3_id = []
    col4_id = []

    for i in nlp_docs:
        for y in nlp_docs:
            col1.append(i)
            col2.append(y)

    for i in df["content"]:
        for y in df["content"]:
            col3.append(i)
            col4.append(y)

    for i in df["title"]:
        for y in df["title"]:
            col3_title.append(i)
            col4_title.append(y)

    for i in df["id"]:
        for y in df["id"]:
            col3_id.append(i)
            col4_id.append(y)

    nlp_df = pd.DataFrame({"col1": col1, "col2": col2, "col3": col3,
                           "col4": col4, "col3_title": col3_title, "col4_title": col4_title,
                           "col3_id": col3_id, "col4_id": col4_id})

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

nlp_df = model_pipeline(df, old_df, nlp)

def news_recommender(nlp_df, new_title, rec_count=10):
    rec_df = nlp_df[nlp_df['col3_title'].str.contains(new_title) == True].sort_values(by="Similarity_Scores", ascending=False)[
             0:rec_count]
    print(pd.DataFrame(rec_df[["col4_id", "col4_title"]]))

new_title = "NASA Scrubs Second Launch Attempt of Artemi"
news_recommender(nlp_df, new_title)