import helper
import embeddings
import splitting
import filtering
import preprocessData
import storingAndLoading
from anytree import Node
from fuzzywuzzy import process
from top2vec import Top2Vec
import top2vec_baseline


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

    Place_Sentences, Person_Sentences, Content_Sentences, Day_Sentences, Month_Sentences, Year_Sentences, Date_Sentences, docs_dict, title_dict, text_dict, ner_dict, pos_dict, token_dict, unique_ner_list, unique_pos_list, unique_token_list_full = helper.get_sentences_from_news(
        df, news_content, news_publisher_title, title, news_content_WO_preprocssing)
    cluster_embeddings_dict_full = embeddings.get_cluster_embeddings_full(model_name, Place_Sentences, Person_Sentences,
                                                                          Content_Sentences, Day_Sentences,
                                                                          Month_Sentences,
                                                                          Year_Sentences, title, umap_flag, umap_dict, token_dict, unique_ner_list, unique_pos_list, unique_token_list_full)
    # docs_dict, title_dict, text_dict, ner_dict, pos_dict = helper.get_doc_ids_text_ner_from_cluster(
    #     news_publisher_title, title,
    #     news_content_WO_preprocssing)

    top2vec_model = Top2Vec(documents=news_content, speed="learn", workers=8)

    storingAndLoading.storeData(Place_Sentences, Person_Sentences, Content_Sentences, Day_Sentences, Month_Sentences,
                                Year_Sentences, Date_Sentences,
                                cluster_embeddings_dict_full, docs_dict, title_dict, text_dict, ner_dict, pos_dict,
                                len(news_content), top2vec_model)

    top2vec_baseline.run_Top2Vec()
    ui_parameters = storingAndLoading.load_ui_parameters()
    run_WEHONA(ui_parameters["split_entity_list_fromUI"], 1000,
               ui_parameters["content_capture_needed"], ui_parameters["time_place_weight"],
               ui_parameters["content_weight"], ui_parameters["topic_interest_keyword"],
               ui_parameters["from_date_keyword"], ui_parameters["to_date_keyword"], 95)


def generateHierarchy(split_entity_list_fromUI, content_depth_needed, content_capture_needed, time_place_weight,
                      content_weight, topic_interest_keyword, from_date_keyword, to_date_keyword, cluster_method):
    ratio_limit = 95
    ui_parameters = storingAndLoading.load_ui_parameters()

    if cluster_method == "Hubble":
        if ui_parameters["split_entity_list_fromUI"] != split_entity_list_fromUI or ui_parameters[
            "content_capture_needed"] != content_capture_needed or ui_parameters[
            "time_place_weight"] != time_place_weight or ui_parameters["content_weight"] != content_weight or \
                ui_parameters["topic_interest_keyword"] != topic_interest_keyword or ui_parameters[
            "from_date_keyword"] != from_date_keyword or ui_parameters["to_date_keyword"] != to_date_keyword:
            run_WEHONA(split_entity_list_fromUI, 10, content_capture_needed, time_place_weight, content_weight,
                       topic_interest_keyword, from_date_keyword, to_date_keyword, ratio_limit)
        elif ui_parameters["content_depth_needed"] != content_depth_needed:
            alter_WEHONA(content_depth_needed)
    elif cluster_method == "Voyager":
        if ui_parameters["content_depth_needed"] != content_depth_needed:
            alter_Top2Vec(content_depth_needed)

    ui_parameters["split_entity_list_fromUI"] = list(dict.fromkeys(split_entity_list_fromUI))
    ui_parameters["content_depth_needed"] = content_depth_needed
    ui_parameters["content_capture_needed"] = content_capture_needed
    ui_parameters["time_place_weight"] = time_place_weight
    ui_parameters["content_weight"] = content_weight
    ui_parameters["topic_interest_keyword"] = topic_interest_keyword
    ui_parameters["from_date_keyword"] = from_date_keyword
    ui_parameters["to_date_keyword"] = to_date_keyword

    storingAndLoading.store_ui_parameters(ui_parameters)


def search_node(search_term):
    cluster_name_dict = storingAndLoading.dynamic_load_cluster_name_dict_news()  # check for top2vec or wehona
    options = list(cluster_name_dict.values())
    highest = process.extractOne(search_term, options)
    cluster_label = {k for k, v in cluster_name_dict.items() if v == highest[0]}
    return "".join(cluster_label).replace("cluster_", "")


def run_WEHONA(split_entity_list_fromUI, content_depth_needed, content_capture_needed, time_place_weight,
               content_weight, topic_interest_keyword, from_date_keyword, to_date_keyword, ratio_limit):
    storingAndLoading.store_summaries({})
    cluster_info_for_not_clustered_data_dict = {}
    Nodes_dict = {}
    entity_naming_dict = {}
    content_depth_now = 1
    Place_Sentences, Person_Sentences, Content_Sentences, Day_Sentences, Month_Sentences, Year_Sentences, \
    Date_Sentences, cluster_embeddings_dict_full, docs_dict, title_dict, text_dict, ner_dict, pos_dict, weights, news_content_length, top2vec_model = storingAndLoading.loadData()
    parent_cluster_main_phase_1 = Node([x for x in range(news_content_length)])
    clusters_to_furthur_split = helper.fetchDocumentstoSplit(text_dict, Date_Sentences, topic_interest_keyword,
                                                             from_date_keyword, to_date_keyword,
                                                             news_content_length)
    if not clusters_to_furthur_split:
        raise Exception("Unable to find documents for the given filters")
    weights, possible_content_depth = helper.create_content_weights(len(clusters_to_furthur_split), weights,
                                                                    content_capture_needed)
    if content_depth_needed > possible_content_depth:  # set max value in ui
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
            cluster_embeddings_dict_full, weights, entity_name, clusters_to_furthur_split, nodes_edges_main,
            Nodes_dict,
            child_count, cluster_info_for_not_clustered_data_dict, entity_naming_dict, content_depth_now,
            time_place_weight, content_weight)
    storingAndLoading.storeUseFlat({"useFlat": True})
    nodes_edges_main['docs_dict'], nodes_edges_main['text_dict'] = docs_dict, text_dict

    possible_content_depth_nodes_edges = 0
    for node in nodes_edges_main['nodes']:
        if node["level"] > possible_content_depth_nodes_edges:
            possible_content_depth_nodes_edges = node["level"]

    nodes_edges_main['possible_content_depth'] = possible_content_depth_nodes_edges

    nodes_edges_main = filtering.eventRepresentation(nodes_edges_main, title_dict, text_dict, Place_Sentences,
                                                     Person_Sentences,
                                                     Date_Sentences, ratio_limit)
    nodes_edges_main, cluster_name_dict = filtering.filter_nodes_edges(nodes_edges_main, ner_dict, pos_dict,
                                                                       ratio_limit)

    nodes_edges_main = helper.find_related_events(nodes_edges_main, cluster_embeddings_dict_full)


    storingAndLoading.dynamic_store_cluster_name_dict_news(cluster_name_dict)
    storingAndLoading.static_store_cluster_name_dict_news(cluster_name_dict)
    storingAndLoading.storeDynamicNews(nodes_edges_main)
    storingAndLoading.storeStaticNews(nodes_edges_main)


def alter_WEHONA(content_depth_needed):
    static_news = storingAndLoading.load_static_news()
    static_search = storingAndLoading.static_load_cluster_name_dict_news()
    nodes = static_news["nodes"]
    nodes_updated = []
    search_updated = {}
    for node in nodes:
        if node["level"] <= content_depth_needed:
            nodes_updated.append(node)
            cluster_number = "cluster_" + str(node["id"])
            search_updated[cluster_number] = static_search[cluster_number]
    static_news["nodes"] = nodes_updated
    storingAndLoading.storeDynamicNews(static_news)
    storingAndLoading.dynamic_store_cluster_name_dict_news(search_updated)


def alter_Top2Vec(content_depth_needed):
    static_top2vec_news = storingAndLoading.load_static_top2vec_news()
    static_top2vec_search = storingAndLoading.static_load_cluster_name_dict_top2vec()
    nodes = static_top2vec_news["nodes"]
    nodes_updated = []
    search_updated = {}
    for node in nodes:
        if node["level"] <= content_depth_needed:
            nodes_updated.append(node)
            cluster_number = "cluster_" + str(node["id"])
            search_updated[cluster_number] = static_top2vec_search[cluster_number]
    static_top2vec_news["nodes"] = nodes_updated
    storingAndLoading.storeDynamictop2vecNews(static_top2vec_news)
    storingAndLoading.dynamic_store_cluster_name_dict_top2vec(search_updated)


# def run_nothing():
#     print("hi")


def generate_custer_summary(cluster_method_no):
    cluster_method_no_list = cluster_method_no.split(":")
    cluster_method = cluster_method_no_list[0]
    cluster_no = cluster_method_no_list[1]
    summaries = storingAndLoading.load_summaries()
    if cluster_no in summaries:
        return summaries[cluster_no]
    else:
        if cluster_method == "Hubble":
            news = storingAndLoading.load_dynamic_news()
        elif cluster_method == "Voyager":
            news = storingAndLoading.load_dynamic_top2vec_news()
        cluster_summary = helper.generate_custer_summary(news, cluster_no)
        summaries[cluster_no] = cluster_summary
        storingAndLoading.store_summaries(summaries)
        return cluster_summary
