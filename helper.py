import spacy
from num2words import num2words
import anytree
from anytree import LevelOrderGroupIter
from scipy.stats import halfnorm
import numpy as np
import math
from datetime import datetime
import pandas as pd

nlp = spacy.load('en_core_web_md')


def get_sentences_from_news(df, news_content):
    print("creating sentences from data ...")
    Place_Sentences = []
    Person_Sentences = []
    Content_Sentences = []
    for news in news_content:
        place_sent = ""
        person_sent = ""
        doc = nlp(news)
        for ent in doc.ents:
            if len(ent.text) > 2:
                if ent.label_ == "PERSON":
                    person_sent = person_sent + ent.text + ","
                elif ent.label_ == "GPE":
                    place_sent = place_sent + ent.text + ","
        Place_Sentences.append(place_sent.strip(','))
        Person_Sentences.append(person_sent.strip(","))
        Content_Sentences.append(news)
    Day_Sentences = df['day'].apply(lambda text: num2words(text)).tolist()
    Month_Sentences = df['month'].tolist()
    Year_Sentences = df['year'].apply(lambda text: num2words(text).replace(' and ', ' ')).tolist()
    Date_Sentences = df['date'].tolist()
    return Place_Sentences, Person_Sentences, Content_Sentences, Day_Sentences, Month_Sentences, Year_Sentences, Date_Sentences


def create_nodes_edges_from_hierarchy(parent_cluster_main, parent_count, child_count, level_number, entity_name_list,
                                      entity_naming_dict):
    cluster_dict = {}
    nodes_list = []
    edges_list = []
    id_dict = {}
    levels = [[node.name for node in children] for children in LevelOrderGroupIter(parent_cluster_main)]
    if parent_count == 0:
        nodes_list.append(
            {'id': parent_count, 'label': "cluster_" + str(parent_count), 'level': level_number, "shape": "box"})
        id_dict[str(levels[0][0])] = parent_count
        cluster_dict["cluster_" + str(parent_count)] = levels[0][0]
    else:
        id_dict[str(levels[0][0])] = parent_count - 1
    for level, parent in enumerate(levels[1:]):
        for child in parent:
            label = "cluster_" + str(child_count)
            nodes_list.append({'id': child_count, 'label': label, 'level': level_number + level + 1, "shape": "box"})
            id_dict[str(child)] = child_count
            child_count = child_count + 1
            cluster_dict[label] = list(child)
    for level, parent in enumerate(levels[1:]):
        for child in parent:
            path = anytree.search.findall_by_attr(parent_cluster_main, child)
            try:
                parent_for_child = str(path[0]).split("/")[-2]
                parent_id = id_dict[parent_for_child]
            except:
                # can use continue here need to check
                # continue
                parent_for_child = str(child)
                parent_id = id_dict[parent_for_child]
            if entity_naming_dict != 'every_node_content':
                label = entity_naming_dict[str(child)]
            else:
                label = 'content'
            child_id = id_dict[str(child)]
            edges_list.append({'from': parent_id, 'to': child_id, 'label': label, 'font': {'align': 'middle'}})
    nodes_edges = {"nodes": nodes_list, "edges": edges_list, "cluster_dict": cluster_dict}
    return nodes_edges, child_count


def get_doc_ids_text_ner_from_cluster(news_publisher_title, title, news_content):
    print("fetching doc ids, text, ner lists form data ...")
    docs_dict = {}
    title_dict = {}
    text_dict = {}
    ner_dict = {}
    pos_dict = {}
    stopwords_df = pd.read_csv("resources/sw1k.csv")
    news_stopwords_list = stopwords_df["term"].tolist()
    for k in range(len(news_publisher_title)):
        title_dict[k] = title[k]
        docs_dict[k] = news_publisher_title[k]
        text_dict[k] = news_content[k]
        ner_sent = ""
        ner_sent_full = ""
        pos_sent = ""
        doc = nlp(news_content[k])
        for ent in doc.ents:
            ner_sent_full = ner_sent_full + ent.text + " "
            if len(ent.text) > 2 and ent.label_ not in ["CARDINAL", "DATE", "GPE", "LANGUAGE", "LOC", "MONEY",
                                                        "ORDINAL", "PERCENT", "PERSON",
                                                        "TIME"] and ent.text.lower() not in news_stopwords_list:
                ner_sent = ner_sent + ent.text + " : "
        ner_dict[k] = ner_sent.strip(" : ").split(" : ")
        for token in doc:
            if token.pos_ in ["NOUN", "PROPN"] and token.is_stop == False and token.text not in ner_sent_full and token.text.lower() not in news_stopwords_list:
                pos_sent = pos_sent + token.text + " : "
        pos_dict[k] = pos_sent.strip(" : ").split(" : ")
    return docs_dict, title_dict, text_dict, ner_dict, pos_dict


def getSplitEntityList(split_entity_list_fromUI, content_depth_needed):
    if "content" in split_entity_list_fromUI:
        content_index = split_entity_list_fromUI.index('content') + 1
        for i in range(content_depth_needed - 1):
            split_entity_list_fromUI[content_index:content_index] = ['content']
            content_index = content_index + 1
    return split_entity_list_fromUI


def create_content_weights(no_of_docs, weights, content_capture_needed):
    mean = 0

    weights_parameters_list = []
    content_depth_needed = 100
    mul = no_of_docs / 10
    data = np.arange(0, content_depth_needed, 1)
    probdf = halfnorm.pdf(data, loc=mean, scale=content_capture_needed)
    div = (math.ceil(probdf[0] * 1000))
    previous = 0
    for i in probdf:
        now = math.ceil((((math.ceil(i * 1000)) * mul) / div))
        if now == previous:
            break
        else:
            previous = now
            weights_parameters_list.append(now + 1)

    content_dict = {}
    for index, min_size in enumerate(weights_parameters_list):
        content_dict[str(index + 1)] = [[0, 0, 0, 0.2, 0.8, 0, 0, 0, 1],
                                        {"min_cluster_size": min_size,
                                         "min_samples": min_size,
                                         "allow_single_cluster": False,
                                         "cluster_selection_epsilon": 0,
                                         "cluster_selection_method": "eom"}]

    weights["content"] = content_dict
    print("Setting content weights ... ", weights_parameters_list)
    return weights, len(weights_parameters_list)


def fetchDocumentstoSplit(text_dict, Date_Sentences, topic_interest_keyword, from_date_keyword, to_date_keyword,
                          news_content_length):
    clusters_to_split_by_key = []
    clusters_to_split_by_date = []
    if topic_interest_keyword.strip() == "":
        clusters_to_split_by_key = [x for x in range(news_content_length)]
    else:
        topic_interest_keyword = " " + topic_interest_keyword.lower().strip() + " "
        for key, value in text_dict.items():
            if value.lower().count(topic_interest_keyword) > 0:
                clusters_to_split_by_key.append(key)
    if from_date_keyword == "":
        from_date_keyword = "0001-01-01 00:00:00"
    else:
        from_date_keyword = from_date_keyword + " 00:00:00"
    if to_date_keyword == "":
        to_date_keyword = "9999-01-01 00:00:00"
    else:
        to_date_keyword = to_date_keyword + " 00:00:00"
    from_date = datetime.strptime(from_date_keyword, "%Y-%m-%d %H:%M:%S")
    to_date = datetime.strptime(to_date_keyword, "%Y-%m-%d %H:%M:%S")
    for index, date in enumerate(Date_Sentences):
        if from_date <= date <= to_date:
            clusters_to_split_by_date.append(index)
    clusters_to_split = list(set(clusters_to_split_by_key).intersection(clusters_to_split_by_date))
    return clusters_to_split
