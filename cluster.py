import helper
import embeddings
import splitting
import filtering
import preprocessData
import storingAndLoading
from anytree import Node
from fuzzywuzzy import process


# import warnings
# warnings.filterwarnings("ignore")


def storeHierarchyData():
    model_name = 'paraphrase-MiniLM-L3-v2'
    umap_flag = False
    umap_dict = {'n_neighbors': 30, 'min_dist': 0.0, 'n_components': 5, 'random_state': 42}
    df = preprocessData.get_news_data()
    news_content_WO_preprocssing = [x.replace('\\', '') for x in df["main_content"].tolist()]
    df = preprocessData.get_preprocessed_data(df)

    news_content = [x.replace('\\', '') for x in df["main_content"].tolist()]
    news_publisher_title = [x.replace('\\', '') for x in df["publisher_title"].tolist()]
    title = [x.replace('\\', '') for x in df["title"].tolist()]

    Place_Sentences, Person_Sentences, Content_Sentences, Day_Sentences, Month_Sentences, Year_Sentences, Date_Sentences = helper.get_sentences_from_news(
        df, news_content)
    cluster_embeddings_dict_full = embeddings.get_cluster_embeddings_full(model_name, Place_Sentences, Person_Sentences,
                                                                          Content_Sentences, Day_Sentences,
                                                                          Month_Sentences,
                                                                          Year_Sentences, umap_flag, umap_dict)
    docs_dict, title_dict, text_dict, ner_dict, pos_dict = helper.get_doc_ids_text_ner_from_cluster(
        news_publisher_title, title,
        news_content_WO_preprocssing)

    storingAndLoading.storeData(Place_Sentences, Person_Sentences, Content_Sentences, Day_Sentences, Month_Sentences,
                                Year_Sentences, Date_Sentences,
                                cluster_embeddings_dict_full, docs_dict, title_dict, text_dict, ner_dict, pos_dict,
                                len(news_content))


def generateHierarchy(split_entity_list_fromUI, content_depth_needed, content_capture_needed, time_place_weight,
                      content_weight, topic_interest_keyword, from_date_keyword, to_date_keyword):
    cluster_info_for_not_clustered_data_dict = {}
    Nodes_dict = {}
    entity_naming_dict = {}
    ratio_limit = 95
    content_depth_now = 1

    Place_Sentences, Person_Sentences, Content_Sentences, Day_Sentences, Month_Sentences, Year_Sentences, \
    Date_Sentences, cluster_embeddings_dict_full, docs_dict, title_dict, text_dict, ner_dict, pos_dict, weights, news_content_length = storingAndLoading.loadData()

    parent_cluster_main_phase_1 = Node([x for x in range(news_content_length)])
    clusters_to_furthur_split = helper.fetchDocumentstoSplit(text_dict, Date_Sentences, topic_interest_keyword, from_date_keyword, to_date_keyword, news_content_length)
    if not clusters_to_furthur_split:
        raise Exception("Unable to find documents for the given filters")

    weights, possible_content_depth = helper.create_content_weights(len(clusters_to_furthur_split), weights,
                                                                    content_capture_needed)

    if content_depth_needed > possible_content_depth:
        content_depth_needed = possible_content_depth

    split_entity_list = helper.getSplitEntityList(split_entity_list_fromUI, content_depth_needed)

    print("splitting data and generating nodes and edges...")
    parent_cluster_main_phase_1, cluster_info_for_not_clustered_data_dict, clusters_to_furthur_split, Nodes_dict, ids_based_on_labels, entity_name_list, entity_naming_dict, content_depth_now = splitting.split_for_3_levels(
        cluster_embeddings_dict_full, split_entity_list[0], weights, parent_cluster_main_phase_1,
        clusters_to_furthur_split, cluster_info_for_not_clustered_data_dict, Nodes_dict, entity_naming_dict, True,
        content_depth_now, time_place_weight, content_weight)
    nodes_edges_main, child_count = helper.create_nodes_edges_from_hierarchy(parent_cluster_main_phase_1, 0, 1, 0,
                                                                             entity_name_list, entity_naming_dict)
    for entity_name in split_entity_list[1:]:
        nodes_edges_main, child_count, clusters_to_furthur_split, Nodes_dict, content_depth_now = splitting.perform_furthur_split_by_entity(
            cluster_embeddings_dict_full, weights, entity_name, clusters_to_furthur_split, nodes_edges_main, Nodes_dict,
            child_count, cluster_info_for_not_clustered_data_dict, entity_naming_dict, content_depth_now,
            time_place_weight, content_weight)

    storingAndLoading.storeUseFlat({"useFlat": True})

    nodes_edges_main['docs_dict'], nodes_edges_main['text_dict'] = docs_dict, text_dict
    nodes_edges_main['possible_content_depth'] = possible_content_depth

    nodes_edges_main = filtering.eventRepresentation(nodes_edges_main, title_dict, text_dict, Place_Sentences, Person_Sentences,
                                                     Date_Sentences, ratio_limit)
    nodes_edges_main, cluster_name_dict = filtering.filter_nodes_edges(nodes_edges_main, ner_dict, pos_dict,
                                                                       ratio_limit)
    storingAndLoading.store_cluster_name_dict(cluster_name_dict)
    storingAndLoading.storeNews(nodes_edges_main)


def search_node(search_term):
    cluster_name_dict = storingAndLoading.load_cluster_name_dict()
    options = list(cluster_name_dict.values())
    highest = process.extractOne(search_term, options)
    cluster_label = {k for k, v in cluster_name_dict.items() if v == highest[0]}
    return "".join(cluster_label).replace("cluster_", "")
