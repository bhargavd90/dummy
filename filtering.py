import numpy as np
from fuzzywuzzy import fuzz
from collections import Counter
import textwrap
import storingAndLoading
from fuzzywuzzy.process import dedupe
import spacy
import pytextrank

nlp = spacy.load("en_core_web_md")
nlp.add_pipe("textrank")


def get_unique_list_based_on_fuzzy_matching(token_list, ratio_limit):
    # uniques = []
    # for key_word in ner_list:
    #     if not uniques:
    #         uniques.append(key_word)
    #     for unique in uniques:
    #         if fuzz.partial_ratio(unique.lower().strip(), key_word.lower().strip()) > ratio_limit:
    #             break
    #     else:
    #         uniques.append(key_word)
    uniques = list(dedupe(token_list, threshold=ratio_limit, scorer=fuzz.partial_ratio))
    return uniques


def get_cluster_words(index, cluster_no, cluster_dict, ner_dict, pos_dict, ratio_limit, use_pos):
    if use_pos:
        keyword_dict = pos_dict
    else:
        keyword_dict = ner_dict
    if index == 0:
        return "News Articles", ""
    else:
        cluster_name_list = []
        keyword_list_full = []
        for doc_id in cluster_dict[cluster_no]:
            keyword_list_full = keyword_list_full + keyword_dict[doc_id]
        ctr_2 = Counter(keyword_list_full)
        most_common_keyword = ctr_2.most_common(50)
        for clus_nm in most_common_keyword:
            cluster_name_list.append(clus_nm[0])
        cluster_name_unique_list = get_unique_list_based_on_fuzzy_matching(cluster_name_list, ratio_limit)
        cluster_name = ",".join(cluster_name_unique_list[0:5])
        cluster_name_search = " ".join(cluster_name_unique_list[0:5]).lower()
        return cluster_name, cluster_name_search


def find_from_by_to(edges_dict_updated, to):
    for edge in edges_dict_updated:
        if edge["to"] == to:
            return edge["from"]


def find_node_by_id(nodes_dict_updated, e_from):
    for node in nodes_dict_updated:
        if node["id"] == e_from:
            return node["colorDict_id"]


def filter_nodes_edges(nodes_edges_main, ner_dict, pos_dict, ratio_limit):
    print("filtering nodes and edges...")

    colorsDict = storingAndLoading.loadColors()
    cluster_name_dict = {}

    nodes_dict = nodes_edges_main['nodes']
    edges_dict = nodes_edges_main['edges']
    cluster_dict = nodes_edges_main['cluster_dict']
    edges_dict_updated = edges_dict
    edges_dict_from = []
    edges_dict_to = []
    edges_filter_list_2 = []
    for edge in edges_dict:
        if edge['from'] == edge['to']:
            edges_dict_updated.remove(edge)
    for edge in edges_dict_updated:
        edges_dict_from.append(edge['from'])
        edges_dict_to.append(edge['to'])
    edges_filter_list_1 = list(set(edges_dict_from + edges_dict_to))
    for from_edge in edges_dict_from:
        if edges_dict_from.count(from_edge) == 1:
            edges_filter_list_2.append(from_edge)
    nodes_to_remove = edges_filter_list_2
    if len(edges_filter_list_1) > 0:
        if 0 in edges_filter_list_1:
            nodes_to_remove = nodes_to_remove + list(
                set(range(edges_filter_list_1[len(edges_filter_list_1) - 1])[1:]) - set(edges_filter_list_1))
        else:
            nodes_to_remove = nodes_to_remove + [0] + list(
                set(range(edges_filter_list_1[len(edges_filter_list_1) - 1])[1:]) - set(edges_filter_list_1))
    nodes_dict_updated = list(np.delete(nodes_dict, nodes_to_remove))
    nodes_edges_main['edges'] = edges_dict_updated
    for node_to_remove in nodes_to_remove:
        cluster_dict.pop(list(cluster_dict.keys())[node_to_remove])

    nodes_dict_updated_temp = nodes_dict_updated

    colorDict_id = 1
    for index, node in enumerate(nodes_dict_updated_temp):
        cluster_no = node["label"]

        if int(cluster_no.replace("cluster_", "")) in edges_dict_from:
            use_pos = True
        else:
            use_pos = False

        cluster_name, cluster_name_search = get_cluster_words(index, cluster_no, cluster_dict, ner_dict, pos_dict,
                                                              ratio_limit, use_pos)
        cluster_name_dict[cluster_no] = cluster_name_search

        nodes_dict_updated[index]["label"] = "\n".join(textwrap.wrap(cluster_name, 15))

        if index != 0:
            e_to = nodes_dict_updated[index]["id"]
            e_from = find_from_by_to(edges_dict_updated, e_to)
            try:
                colorDict_id_from_node = find_node_by_id(nodes_dict_updated, e_from)
            except:
                colorDict_id_from_node = 0

            # patch work
            if colorDict_id_from_node is None:
                nodes_dict_updated.pop(1)
                break

            level_no = nodes_dict_updated[index]["level"]

            if level_no > 5:
                level_no = 5

            if colorDict_id_from_node == 0:
                nodes_dict_updated[index]["color"] = {"background": colorsDict[str(colorDict_id)][str(level_no)],
                                                      "border": "black"}
                nodes_dict_updated[index]["colorDict_id"] = colorDict_id
                colorDict_id = colorDict_id + 1
                if colorDict_id == 11:
                    colorDict_id = 1
            else:
                nodes_dict_updated[index]["color"] = {
                    "background": colorsDict[str(colorDict_id_from_node)][str(level_no)], "border": "black"}
                nodes_dict_updated[index]["colorDict_id"] = colorDict_id_from_node

    nodes_dict_updated[0]["colorDict_id"] = 0
    nodes_edges_main['nodes'] = nodes_dict_updated
    return nodes_edges_main, cluster_name_dict


def eventRepresentation(nodes_edges_main, title_dict, text_dict, Place_Sentences, Person_Sentences, Date_Sentences,
                        ratio_limit):
    Place_dict = {}
    Person_dict = {}
    Date_dict = {}
    Title_dict = {}
    Summary_dict = {}
    for cluster_name, cluster_ids in nodes_edges_main["cluster_dict"].items():
        Place_Sentences_list = []
        Person_Sentences_list = []
        Date_Sentences_list = []
        # Title_Sentence_all = ""
        # Summary_Sentence_all = ""
        # Title_Sentence = ""
        # Summary_Sentence = ""
        for index, doc_id in enumerate(cluster_ids):
            Place_Sentences_list = Place_Sentences_list + Place_Sentences[doc_id].split(",")
            Person_Sentences_list = Person_Sentences_list + Person_Sentences[doc_id].split(",")
            Date_Sentences_list.append(str(Date_Sentences[doc_id]).replace(" 00:00:00", ""))
            if index == 0:
                Title_Sentence = title_dict[doc_id]
            # if index < 10:
            #     Summary_Sentence_all = Summary_Sentence_all + text_dict[doc_id] + ". "

        place_words = list(filter(None, list(
            dict.fromkeys(sorted(Place_Sentences_list, key=Counter(Place_Sentences_list).get, reverse=True)))[0:50]))
        person_words = list(filter(None, list(
            dict.fromkeys(sorted(Person_Sentences_list, key=Counter(Person_Sentences_list).get, reverse=True)))[0:50]))
        date_words = list(filter(None, list(
            dict.fromkeys(sorted(Date_Sentences_list, key=Counter(Date_Sentences_list).get, reverse=True)))[0:50]))
        Place_dict["Place_" + cluster_name] = get_unique_list_based_on_fuzzy_matching(place_words, ratio_limit)[0:10]
        Person_dict["Person_" + cluster_name] = get_unique_list_based_on_fuzzy_matching(person_words, ratio_limit)[0:10]
        Date_dict["Date_" + cluster_name] = get_unique_list_based_on_fuzzy_matching(date_words, ratio_limit)[0:10]

        # titleDoc = nlp(Title_Sentence_all.strip(". "))
        # for title_sent in titleDoc._.textrank.title_summary(limit_phrases=100, limit_sentences=1):
        #     Title_Sentence = Title_Sentence + str(title_sent) + ". "
        # summaryDoc = nlp(Summary_Sentence_all.strip(". "))
        # for summary_sent in summaryDoc._.textrank.title_summary(limit_phrases=100, limit_sentences=5):
        #     Summary_Sentence = Summary_Sentence + str(summary_sent) + ". "

        Title_dict["Title_" + cluster_name] = Title_Sentence
        Summary_dict["Summary_" + cluster_name] = ""

    nodes_edges_main["Place_dict"] = Place_dict
    nodes_edges_main["Person_dict"] = Person_dict
    nodes_edges_main["Date_dict"] = Date_dict
    nodes_edges_main["Title_dict"] = Title_dict
    nodes_edges_main["Summary_dict"] = Summary_dict
    return nodes_edges_main
