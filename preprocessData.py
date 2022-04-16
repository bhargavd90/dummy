import pandas as pd
from datetime import datetime
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re


def get_news_data():
    print("fetching data ...")
    df_original = pd.read_parquet("datasets/uci_news.snappy.parquet")
    df_toy = pd.read_csv("datasets/Toy_Dataset_Thesis - Sheet1.csv", header=None)
    df_original = df_original[['title', 'publisher', 'main_content', 'timestamp', 'category']]
    df_original = df_original.replace(["b", "t", "e", "m"], ["Business", "Science & Technology", "Entertainment", "Health"])
    toy_dataset_ids = []
    for _, row in df_toy.iterrows():
        ids_string = row[1]
        ids_list = ids_string.split(",")
        for id in ids_list:
            toy_dataset_ids.append(int(id.strip()))
    print(len(toy_dataset_ids))
    df_original = df_original[df_original.index.isin(toy_dataset_ids)]
    df_original.reset_index(inplace=True)

    # df_original.reset_index(inplace=True)
    # df1 = df_original[['title', 'publisher', 'main_content', 'timestamp']][0:200]
    # df2 = df_original[['title', 'publisher', 'main_content', 'timestamp']][1000:1200]
    # df3 = df_original[['title', 'publisher', 'main_content', 'timestamp']][2000:2200]
    # df4 = df_original[['title', 'publisher', 'main_content', 'timestamp']][3000:3200]
    # df5 = df_original[['title', 'publisher', 'main_content', 'timestamp']][4000:4200]
    # df = pd.concat([df1, df2, df3, df4, df5])

    df = df_original
    df["publisher_title"] = df["publisher"] + ' - ' + df["title"]
    return df


def add_date_to_text(text):
    dt_object = datetime.fromtimestamp(text / 1000).strftime('%Y-%m-%d')
    return dt_object


def add_time_to_text(text):
    dt_object = datetime.fromtimestamp(text / 1000).strftime("%m-%d-%Y, %H:%M:%S")
    return dt_object


def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)


def remove_punctuation(text):
    punct_to_remove = list(string.punctuation)
    punct_to_remove.remove(".")
    for punct in punct_to_remove:
        text = text.replace(punct, " ")
    return text


def stem_words(text):
    stemmer = PorterStemmer()
    return " ".join([stemmer.stem(word) for word in text.split()])


def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    return " ".join([word for word in str(text).split() if word not in stop_words])


def get_preprocessed_data(df):
    print("preprocessing data ...")
    # df["main_content"] = df["main_content"].str.lower()
    df['time'] = df["timestamp"].apply(lambda text: add_time_to_text(text))
    df['date'] = df["timestamp"].apply(lambda text: add_date_to_text(text))
    df['date'] = df["date"].apply(pd.to_datetime)
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.strftime("%B")
    df['year'] = df['date'].dt.year
    df["main_content"] = df["main_content"].apply(lambda text: remove_urls(text))
    df["main_content"] = df["main_content"].apply(lambda text: remove_punctuation(text))
    # df["main_content"] = df["main_content"].apply(lambda text: remove_stopwords(text))
    # df["main_content"] = df["main_content"].apply(lambda text: stem_words(text))
    return df
