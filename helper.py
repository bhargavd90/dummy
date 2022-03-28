import spacy
from num2words import num2words
import anytree
from anytree import LevelOrderGroupIter
from scipy.stats import halfnorm
import numpy as np
import math
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import embeddings
import pytextrank
import torch
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load('en_core_web_md')
nlp.add_pipe("textrank")

model_name = 'paraphrase-MiniLM-L3-v2'
model = SentenceTransformer(model_name)
print('CUDA Device count', torch.cuda.device_count())
if torch.cuda.device_count() >= 1:
    device = torch.device("cuda:0")
    model.to(device)


def get_sentences_from_news(df, news_content, news_publisher_title, title, news_content_WO_preprocssing):
    print("creating sentences from data ...")
    Place_Sentences = []
    Person_Sentences = []
    Content_Sentences = []
    docs_dict = {}
    title_dict = {}
    text_dict = {}
    ner_dict = {}
    pos_dict = {}
    unique_ner_dict = {}
    unique_pos_dict = {}
    stopwords_df = pd.read_csv("resources/sw1k.csv")
    news_stopwords_list = stopwords_df["term"].tolist()
    for k in range(len(news_publisher_title)):
        tokens_list = []
        place_sent = ""
        person_sent = ""
        title_dict[k] = title[k]
        docs_dict[k] = news_publisher_title[k]
        text_dict[k] = news_content_WO_preprocssing[k]
        ner_sent = ""
        ner_sent_full = ""
        pos_sent = ""
        doc = nlp(news_content[k])
        for ent in doc.ents:
            # if len(ent.text) > 2:
            if ent.label_ == "PERSON":
                person_sent = person_sent + ent.text.lower() + ","
            elif ent.label_ == "GPE":
                place_sent = place_sent + ent.text.lower() + ","
            ner_sent_full = ner_sent_full + ent.text.lower() + " "
            if len(ent.text) > 2 and ent.label_ not in ["CARDINAL", "DATE", "GPE", "LANGUAGE", "LOC", "MONEY",
                                                        "ORDINAL", "PERCENT", "PERSON",
                                                        "TIME"] and ent.text.lower() not in news_stopwords_list:
                ner_sent = ner_sent + ent.text.lower() + " : "
        ner_dict[k] = ner_sent.strip(" : ").split(" : ")
        ctr_3 = Counter(ner_dict[k])
        most_common_ner_keyword = ctr_3.most_common(5)
        unique_ner_dict[k] = most_common_ner_keyword
        for token in doc:
            if token.text.lower() not in news_stopwords_list:
                tokens_list.append(token.text.lower())
            if token.pos_ in ["NOUN",
                              "PROPN"] and token.is_stop == False and token.text.lower() not in ner_sent_full and token.text.lower() not in news_stopwords_list:
                pos_sent = pos_sent + token.text.lower() + " : "
        pos_dict[k] = pos_sent.strip(" : ").split(" : ")
        ctr_4 = Counter(pos_dict[k])
        most_common_pos_keyword = ctr_4.most_common(5)
        unique_pos_dict[k] = most_common_pos_keyword
        Place_Sentences.append(place_sent.strip(','))
        Person_Sentences.append(person_sent.strip(","))
        Content_Sentences.append(news_content[k])
    Day_Sentences = df['day'].apply(lambda text: num2words(text)).tolist()
    Month_Sentences = df['month'].tolist()
    Year_Sentences = df['year'].apply(lambda text: num2words(text).replace(' and ', ' ')).tolist()
    Date_Sentences = df['date'].tolist()
    Time_Sentences = df['time'].tolist()
    Category_Sentences = df['category'].tolist()
    return Place_Sentences, Person_Sentences, Content_Sentences, Day_Sentences, Month_Sentences, Year_Sentences, Date_Sentences, Time_Sentences, Category_Sentences, docs_dict, title_dict, text_dict, ner_dict, pos_dict, unique_ner_dict, unique_pos_dict


def create_nodes_edges_from_hierarchy(parent_cluster_main, parent_count, child_count, level_number, entity_name_list,
                                      entity_naming_dict, split_with_size):
    cluster_dict = {}
    nodes_list = []
    edges_list = []
    id_dict = {}
    levels = [[node.name for node in children] for children in LevelOrderGroupIter(parent_cluster_main)]
    if parent_count == 0:
        nodes_list.append(
            {'id': parent_count, 'label': "cluster_" + str(parent_count), 'level': level_number, "shape": "box",
             "split_with_size": split_with_size})
        id_dict[str(levels[0][0])] = parent_count
        cluster_dict["cluster_" + str(parent_count)] = levels[0][0]
    else:
        id_dict[str(levels[0][0])] = parent_count - 1
    for level, parent in enumerate(levels[1:]):
        for child in parent:
            label = "cluster_" + str(child_count)
            nodes_list.append({'id': child_count, 'label': label, 'level': level_number + level + 1, "shape": "box",
                               "split_with_size": split_with_size})
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


# def get_doc_ids_text_ner_from_cluster(news_publisher_title, title, news_content):
#     print("fetching doc ids, text, ner lists form data ...")
#     docs_dict = {}
#     title_dict = {}
#     text_dict = {}
#     ner_dict = {}
#     pos_dict = {}
#     stopwords_df = pd.read_csv("resources/sw1k.csv")
#     news_stopwords_list = stopwords_df["term"].tolist()
#     for k in range(len(news_publisher_title)):
#         title_dict[k] = title[k]
#         docs_dict[k] = news_publisher_title[k]
#         text_dict[k] = news_content[k]
#         ner_sent = ""
#         ner_sent_full = ""
#         pos_sent = ""
#         doc = nlp(news_content[k])
#         for ent in doc.ents:
#             ner_sent_full = ner_sent_full + ent.text + " "
#             if len(ent.text) > 2 and ent.label_ not in ["CARDINAL", "DATE", "GPE", "LANGUAGE", "LOC", "MONEY",
#                                                         "ORDINAL", "PERCENT", "PERSON",
#                                                         "TIME"] and ent.text.lower() not in news_stopwords_list:
#                 ner_sent = ner_sent + ent.text + " : "
#         ner_dict[k] = ner_sent.strip(" : ").split(" : ")
#         for token in doc:
#             if token.pos_ in ["NOUN",
#                               "PROPN"] and token.is_stop == False and token.text not in ner_sent_full and token.text.lower() not in news_stopwords_list:
#                 pos_sent = pos_sent + token.text + " : "
#         pos_dict[k] = pos_sent.strip(" : ").split(" : ")
#     return docs_dict, title_dict, text_dict, ner_dict, pos_dict


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
    mul = no_of_docs / 40
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

    # epsilon_list = [1, 0.5, 0]

    # content_weight = 0
    # pos_weight = 1

    for index, min_size in enumerate(weights_parameters_list):
        # content_weight_temp = content_weight + (index * 0.2)
        # pos_weight_temp = pos_weight - (index * 0.2)
        #
        # if content_weight_temp > 1:
        #     content_weight_temp = 1
        # if pos_weight_temp < 0:
        #     pos_weight_temp = 0

        # if index == 0:
        #     epsilon = epsilon_list[0]
        # elif index == 1:
        #     epsilon = epsilon_list[1]
        # else:
        #     epsilon = epsilon_list[2]

        content_dict[str(index + 1)] = [[0, 0, 0, 0, 1, 0, 0, 0, 1],
                                        {"min_cluster_size": min_size,
                                         # "min_samples": min_size,
                                         "allow_single_cluster": False,
                                         # "cluster_selection_epsilon": epsilon,
                                         "cluster_selection_method": "eom"},
                                        # {"content_weight": content_weight_temp, "pos_weight": pos_weight_temp}
                                        ]

    weights["content"] = content_dict
    print("Setting content weights ... ", weights_parameters_list)
    return weights, len(weights_parameters_list), weights_parameters_list


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


def generate_custer_summary(news, cluster_no):
    text_dict = news["text_dict"]
    doc_ids_in_cluster = news["cluster_dict"][cluster_no]
    total_text_from_cluster = ""
    summary_sentence = "<strong>->  </strong>"
    summary_sentence_list = []
    summary_sentence_dict = {}
    for doc_id in doc_ids_in_cluster[0:10]:
        total_text_from_cluster = total_text_from_cluster + text_dict[str(doc_id)] + ". "
    summaryDoc = nlp(total_text_from_cluster.strip(". "))
    for summary_sent_doc in summaryDoc._.textrank.summary(limit_phrases=100, limit_sentences=50):
        summary_sentence_list.append(summary_sent_doc)

    for index, summary_sent in enumerate(summary_sentence_list):
        if len(summary_sentence_dict) > 4:
            break
        if index == 0:
            summary_sentence_dict[str(summary_sent)] = model.encode(str(summary_sent))
        else:
            flag_list = []
            for summary_sent_key, summary_sent_value in summary_sentence_dict.items():
                doc_emb = model.encode(str(summary_sent))
                sim = round(cosine_similarity([doc_emb], [summary_sent_value]).item(), 3)
                if sim < 0.7:
                    flag_list.append("add")
                else:
                    flag_list.append("delete")
            if not "delete" in flag_list:
                summary_sentence_dict[str(summary_sent)] = doc_emb
    for key in summary_sentence_dict.keys():
        summary_sentence = summary_sentence + key + ". " + "<br>" + "<strong>->  </strong>"
    return summary_sentence.rstrip("<strong>->  </strong>")


def find_related_events(nodes_edges_main, cluster_embeddings_dict_full):
    print("Finding related events ...")
    nodes_list = nodes_edges_main["nodes"]
    nodes_branch_dict = {}
    nodes_color_dict = {}
    related_events_dict = {}
    cluster_centroids_dict = {}
    cluster_centroids_list = []
    cluster_dict = nodes_edges_main["cluster_dict"]
    for node in nodes_list:
        nodes_branch_dict["cluster_" + str(node["id"])] = node["colorDict_id"]
        nodes_color_dict["cluster_" + str(node["id"])] = node["color"]["background"]
    for cluster_no, doc_ids in cluster_dict.items():
        cluster_embeddings_dict_per_cluster = embeddings.get_cluster_embeddings_dict_per_cluster(
            cluster_embeddings_dict_full, doc_ids, {"content_weight": 1, "pos_weight": 0})
        weighted_embeddings = embeddings.get_weighted_embeddings(cluster_embeddings_dict_per_cluster,
                                                                 [0, 0, 0, 0, 1, 0, 0, 0, 1])
        cluster_centroid = np.mean(weighted_embeddings, axis=0)
        cluster_centroids_dict[cluster_no] = cluster_centroid
        cluster_centroids_list.append(cluster_centroid)

    similarities = cosine_similarity(cluster_centroids_list)

    cluster_names = list(cluster_centroids_dict.keys())

    for index, sim_array in enumerate(similarities):
        sim_list = list(sim_array)
        sim_list_sorted = sorted(sim_list, reverse=True)
        for_cluster = cluster_names[index]
        to_cluster_list = []
        count = 0
        for sim_value in sim_list_sorted:
            index_of_highest_similarity = sim_list.index(sim_value)
            to_cluster = cluster_names[index_of_highest_similarity]
            if sim_value == 1 or nodes_branch_dict[for_cluster] == nodes_branch_dict[
                to_cluster] or index_of_highest_similarity == 0:
                continue

            if count > 2 or sim_value < 0.6:
                break
            else:
                to_cluster_list.append([to_cluster, nodes_color_dict[to_cluster]])
                count = count + 1

        related_events_dict[for_cluster] = to_cluster_list
    related_events_dict["cluster_0"] = []
    nodes_edges_main["related_events"] = related_events_dict
    return nodes_edges_main


def create_cluster_match(nodes_edges_main, cluster_embeddings_dict_full):
    cluster_centroids_dict = {}
    content_embeddings = cluster_embeddings_dict_full["content_embeddings"]
    cluster_dict = nodes_edges_main["cluster_dict"]
    for key, value in cluster_dict.items():
        sim_to_center = {}
        cluster_mean = np.mean(content_embeddings[[value]], axis=0).reshape(1, -1)
        for id in value:
            emb = content_embeddings[[id]]
            sim = cosine_similarity(cluster_mean, emb)[0][0]
            sim_to_center[str(id)] = sim
        cluster_centroids_dict[key] = sim_to_center
    nodes_edges_main["cluster_centroids_dict"] = cluster_centroids_dict
    return nodes_edges_main
