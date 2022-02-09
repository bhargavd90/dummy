import pickle
import json
from numpyencoder import NumpyEncoder


def storeData(Place_Sentences, Person_Sentences, Content_Sentences, Day_Sentences, Month_Sentences, Year_Sentences,
              Date_Sentences, cluster_embeddings_dict_full, docs_dict, title_dict, text_dict, ner_dict, pos_dict,
              news_content_length):
    print("storing data ...")
    with open('resources/place_sentences', 'wb') as fp:
        pickle.dump(Place_Sentences, fp)
    with open('resources/person_sentences', 'wb') as fp:
        pickle.dump(Person_Sentences, fp)
    with open('resources/content_sentences', 'wb') as fp:
        pickle.dump(Content_Sentences, fp)
    with open('resources/day_sentences', 'wb') as fp:
        pickle.dump(Day_Sentences, fp)
    with open('resources/month_sentences', 'wb') as fp:
        pickle.dump(Month_Sentences, fp)
    with open('resources/year_sentences', 'wb') as fp:
        pickle.dump(Year_Sentences, fp)
    with open('resources/date_sentences', 'wb') as fp:
        pickle.dump(Date_Sentences, fp)
    with open('resources/cluster_embeddings_dict_full.json', 'wb') as fp:
        pickle.dump(cluster_embeddings_dict_full, fp)
    with open('resources/docs_dict', 'wb') as fp:
        pickle.dump(docs_dict, fp)
    with open('resources/title_dict', 'wb') as fp:
        pickle.dump(title_dict, fp)
    with open('resources/text_dict', 'wb') as fp:
        pickle.dump(text_dict, fp)
    with open('resources/ner_dict', 'wb') as fp:
        pickle.dump(ner_dict, fp)
    with open('resources/pos_dict', 'wb') as fp:
        pickle.dump(pos_dict, fp)
    with open('resources/news_content_length', 'wb') as fp:
        pickle.dump(news_content_length, fp)


def loadData():
    print("loading data ...")
    with open('resources/place_sentences', 'rb') as fp:
        Place_Sentences = pickle.load(fp)
    with open('resources/person_sentences', 'rb') as fp:
        Person_Sentences = pickle.load(fp)
    with open('resources/content_sentences', 'rb') as fp:
        Content_Sentences = pickle.load(fp)
    with open('resources/day_sentences', 'rb') as fp:
        Day_Sentences = pickle.load(fp)
    with open('resources/month_sentences', 'rb') as fp:
        Month_Sentences = pickle.load(fp)
    with open('resources/year_sentences', 'rb') as fp:
        Year_Sentences = pickle.load(fp)
    with open('resources/date_sentences', 'rb') as fp:
        Date_Sentences = pickle.load(fp)
    with open('resources/cluster_embeddings_dict_full.json', 'rb') as fp:
        cluster_embeddings_dict_full = pickle.load(fp)
    with open('resources/docs_dict', 'rb') as fp:
        docs_dict = pickle.load(fp)
    with open('resources/title_dict', 'rb') as fp:
        title_dict = pickle.load(fp)
    with open('resources/text_dict', 'rb') as fp:
        text_dict = pickle.load(fp)
    with open('resources/ner_dict', 'rb') as fp:
        ner_dict = pickle.load(fp)
    with open('resources/pos_dict', 'rb') as fp:
        pos_dict = pickle.load(fp)
    with open('resources/weights.json', 'rb') as fp:
        weights = json.load(fp)
    with open('resources/news_content_length', 'rb') as fp:
        news_content_length = pickle.load(fp)
    return Place_Sentences, Person_Sentences, Content_Sentences, Day_Sentences, Month_Sentences, Year_Sentences, \
           Date_Sentences, cluster_embeddings_dict_full, docs_dict, title_dict, text_dict, ner_dict, pos_dict, weights, news_content_length


def storeNews(nodes_edges_main):
    with open("results/news.json", "w") as outfile:
        json.dump(nodes_edges_main, outfile, cls=NumpyEncoder)


def loadColors():
    with open('resources/colors.json', 'rb') as fp:
        return json.load(fp)


def store_cluster_name_dict(cluster_name_dict):
    with open("results/search.json", "w") as outfile:
        json.dump(cluster_name_dict, outfile, cls=NumpyEncoder)


def load_cluster_name_dict():
    with open('results/search.json', 'rb') as fp:
        return json.load(fp)
