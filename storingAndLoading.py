import pickle
import json
from numpyencoder import NumpyEncoder


def storeData(Place_Sentences, Person_Sentences, Content_Sentences, Day_Sentences, Month_Sentences, Year_Sentences,
              Date_Sentences, Time_Sentences, Category_Sentences, cluster_embeddings_dict_full, docs_dict, title_dict,
              text_dict, ner_dict, pos_dict,
              news_content_length, top2vec_model, category_split):
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
    with open('resources/time_sentences', 'wb') as fp:
        pickle.dump(Time_Sentences, fp)
    with open('resources/category_sentences', 'wb') as fp:
        pickle.dump(Category_Sentences, fp)
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
    with open('resources/top2vec_model', 'wb') as fp:
        pickle.dump(top2vec_model, fp)
    with open('resources/category_split', 'wb') as fp:
        pickle.dump(category_split, fp)


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
    with open('resources/time_sentences', 'rb') as fp:
        Time_Sentences = pickle.load(fp)
    with open('resources/category_sentences', 'rb') as fp:
        Category_Sentences = pickle.load(fp)
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
    with open('resources/top2vec_model', 'rb') as fp:
        top2vec_model = pickle.load(fp)
    with open('resources/category_split', 'rb') as fp:
        category_split = pickle.load(fp)
    return Place_Sentences, Person_Sentences, Content_Sentences, Day_Sentences, Month_Sentences, Year_Sentences, \
           Date_Sentences, Time_Sentences, Category_Sentences, cluster_embeddings_dict_full, docs_dict, title_dict, text_dict, ner_dict, pos_dict, weights, news_content_length, top2vec_model, category_split


def loadColors():
    with open('resources/colors.json', 'rb') as fp:
        return json.load(fp)


def loadUseFlat():
    with open('resources/useFlat.json', 'rb') as fp:
        return json.load(fp)


def storeUseFlat(useFlat):
    with open("resources/useFlat.json", "w") as outfile:
        json.dump(useFlat, outfile, cls=NumpyEncoder)


def load_ui_parameters():
    with open('resources/ui_parameters.json', 'rb') as fp:
        return json.load(fp)


def store_ui_parameters(ui_parameters):
    with open("resources/ui_parameters.json", "w") as outfile:
        json.dump(ui_parameters, outfile, cls=NumpyEncoder)


def dynamic_store_cluster_name_dict_news(cluster_name_dict):
    with open("results_dynamic/search_news.json", "w") as outfile:
        json.dump(cluster_name_dict, outfile, cls=NumpyEncoder)


def static_store_cluster_name_dict_news(cluster_name_dict):
    with open("results_static/search_news.json", "w") as outfile:
        json.dump(cluster_name_dict, outfile, cls=NumpyEncoder)


def static_load_cluster_name_dict_news():
    with open('results_static/search_news.json', 'rb') as fp:
        return json.load(fp)


def dynamic_load_cluster_name_dict_news():
    with open('results_dynamic/search_news.json', 'rb') as fp:
        return json.load(fp)


def dynamic_store_cluster_name_dict_top2vec(cluster_name_dict):
    with open("results_dynamic/search_top2vec.json", "w") as outfile:
        json.dump(cluster_name_dict, outfile, cls=NumpyEncoder)


def static_store_cluster_name_dict_top2vec(cluster_name_dict):
    with open("results_static/search_top2vec.json", "w") as outfile:
        json.dump(cluster_name_dict, outfile, cls=NumpyEncoder)


def static_load_cluster_name_dict_top2vec():
    with open('results_static/search_top2vec.json', 'rb') as fp:
        return json.load(fp)


def dynamic_load_cluster_name_dict_top2vec():
    with open('results_dynamic/search_top2vec.json', 'rb') as fp:
        return json.load(fp)


def load_static_news():
    with open('results_static/news.json', 'rb') as fp:
        return json.load(fp)


def storeDynamicNews(nodes_edges_main):
    with open("results_dynamic/news.json", "w") as outfile:
        json.dump(nodes_edges_main, outfile, cls=NumpyEncoder)


def storeStaticNews(nodes_edges_main):
    with open("results_static/news.json", "w") as outfile:
        json.dump(nodes_edges_main, outfile, cls=NumpyEncoder)


def load_static_top2vec_news():
    with open('results_static/top2vecnews.json', 'rb') as fp:
        return json.load(fp)


def storeDynamictop2vecNews(top2vec_nodes_edges_main):
    with open("results_dynamic/top2vecnews.json", "w") as outfile:
        json.dump(top2vec_nodes_edges_main, outfile, cls=NumpyEncoder)


def storeStatictop2vecNews(top2vec_nodes_edges_main):
    with open("results_static/top2vecnews.json", "w") as outfile:
        json.dump(top2vec_nodes_edges_main, outfile, cls=NumpyEncoder)


def load_dynamic_news():
    with open('results_dynamic/news.json', 'rb') as fp:
        return json.load(fp)


def load_dynamic_top2vec_news():
    with open('results_dynamic/top2vecnews.json', 'rb') as fp:
        return json.load(fp)


def store_summaries_hubble(summaries):
    with open("title_summary_what/summaries_hubble.json", "w") as outfile:
        json.dump(summaries, outfile, cls=NumpyEncoder)


def load_summaries_hubble():
    with open('title_summary_what/summaries_hubble.json', 'rb') as fp:
        return json.load(fp)


def store_whats_hubble(whats):
    with open("title_summary_what/whats_hubble.json", "w") as outfile:
        json.dump(whats, outfile, cls=NumpyEncoder)


def load_whats_hubble():
    with open('title_summary_what/whats_hubble.json', 'rb') as fp:
        return json.load(fp)


def store_summaries_voyager(summaries):
    with open("title_summary_what/summaries_voyager.json", "w") as outfile:
        json.dump(summaries, outfile, cls=NumpyEncoder)


def load_summaries_voyager():
    with open('title_summary_what/summaries_voyager.json', 'rb') as fp:
        return json.load(fp)


def store_whats_voyager(whats):
    with open("title_summary_what/whats_voyager.json", "w") as outfile:
        json.dump(whats, outfile, cls=NumpyEncoder)


def load_whats_voyager():
    with open('title_summary_what/whats_voyager.json', 'rb') as fp:
        return json.load(fp)





def dynamic_store_tfidf_vectorizer_hubble(vectorizer):
    with open('results_dynamic/tfidf_vectorizer_hubble', 'wb') as fp:
        pickle.dump(vectorizer, fp)


def dynamic_store_tfidf_array_hubble(X):
    with open('results_dynamic/tfidf_array_hubble', 'wb') as fp:
        pickle.dump(X, fp)


def dynamic_store_tfidf_vectorizer_voyager(vectorizer):
    with open('results_dynamic/tfidf_vectorizer_voyager', 'wb') as fp:
        pickle.dump(vectorizer, fp)


def dynamic_store_tfidf_array_voyager(X):
    with open('results_dynamic/tfidf_array_voyager', 'wb') as fp:
        pickle.dump(X, fp)


def static_store_tfidf_vectorizer_hubble(vectorizer):
    with open('results_static/tfidf_vectorizer_hubble', 'wb') as fp:
        pickle.dump(vectorizer, fp)


def static_store_tfidf_array_hubble(X):
    with open('results_static/tfidf_array_hubble', 'wb') as fp:
        pickle.dump(X, fp)


def static_store_tfidf_vectorizer_voyager(vectorizer):
    with open('results_static/tfidf_vectorizer_voyager', 'wb') as fp:
        pickle.dump(vectorizer, fp)


def static_store_tfidf_array_voyager(X):
    with open('results_static/tfidf_array_voyager', 'wb') as fp:
        pickle.dump(X, fp)









def dynamic_load_tfidf_vectorizer_hubble():
    with open('results_dynamic/tfidf_vectorizer_hubble', 'rb') as fp:
        return pickle.load(fp)


def dynamic_load_tfidf_array_hubble():
    with open('results_dynamic/tfidf_array_hubble', 'rb') as fp:
        return pickle.load(fp)


def dynamic_load_tfidf_vectorizer_voyager():
    with open('results_dynamic/tfidf_vectorizer_voyager', 'rb') as fp:
        return pickle.load(fp)


def dynamic_load_tfidf_array_voyager():
    with open('results_dynamic/tfidf_array_voyager', 'rb') as fp:
        return pickle.load(fp)


def static_load_tfidf_vectorizer_hubble():
    with open('results_static/tfidf_vectorizer_hubble', 'rb') as fp:
        return pickle.load(fp)


def static_load_tfidf_array_hubble():
    with open('results_static/tfidf_array_hubble', 'rb') as fp:
        return pickle.load(fp)


def static_load_tfidf_vectorizer_voyager():
    with open('results_static/tfidf_vectorizer_voyager', 'rb') as fp:
        return pickle.load(fp)


def static_load_tfidf_array_voyager():
    with open('results_static/tfidf_array_voyager', 'rb') as fp:
        return pickle.load(fp)