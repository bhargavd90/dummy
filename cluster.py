import helper
import embeddings
import splitting
import filtering
import preprocessData
import storingAndLoading
from anytree import Node
#from fuzzywuzzy import process
# from top2vec import Top2Vec
import top2vec_baseline
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score


# import warnings
# warnings.filterwarnings("ignore")


def storeHierarchyData():
    model_name = 'paraphrase-MiniLM-L3-v2'
    umap_flag = False
    umap_dict = {'n_neighbors': 30, 'min_dist': 0.0, 'n_components': 5, 'random_state': 42}
    df = preprocessData.get_news_data()
    news_content_WO_preprocssing = [x.replace('\\', '') for x in df["main_content"].tolist()]
    df = preprocessData.get_preprocessed_data(df)

    category_split = helper.split_by_category(df)

    news_content = [x.replace('\\', '') for x in df["main_content"].tolist()]
    news_publisher_title = [x.replace('\\', '') for x in df["publisher_title"].tolist()]
    title = [x.replace('\\', '') for x in df["title"].tolist()]

    Place_Sentences, Person_Sentences, Content_Sentences, Day_Sentences, Month_Sentences, Year_Sentences, Date_Sentences, Time_Sentences, Category_Sentences, docs_dict, title_dict, text_dict, ner_dict, pos_dict, unique_ner_dict, unique_pos_dict = helper.get_sentences_from_news(
        df, news_content, news_publisher_title, title, news_content_WO_preprocssing)
    cluster_embeddings_dict_full = embeddings.get_cluster_embeddings_full(model_name, Place_Sentences, Person_Sentences,
                                                                          Content_Sentences, Day_Sentences,
                                                                          Month_Sentences,
                                                                          Year_Sentences, title, umap_flag, umap_dict,
                                                                          unique_ner_dict, unique_pos_dict)
    # docs_dict, title_dict, text_dict, ner_dict, pos_dict = helper.get_doc_ids_text_ner_from_cluster(
    #     news_publisher_title, title,
    #     news_content_WO_preprocssing)

    # top2vec_model = Top2Vec(documents=news_content, speed="learn", workers=8)

    top2vec_model = "no model"

    storingAndLoading.storeData(Place_Sentences, Person_Sentences, Content_Sentences, Day_Sentences, Month_Sentences,
                                Year_Sentences, Date_Sentences, Time_Sentences, Category_Sentences,
                                cluster_embeddings_dict_full, docs_dict, title_dict, text_dict, ner_dict, pos_dict,
                                len(news_content), top2vec_model, category_split)

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


def search_node(search_term, method_name):
    if method_name == "Hubble":
        vectorizer = storingAndLoading.dynamic_load_tfidf_vectorizer_hubble()
        X_dict = storingAndLoading.dynamic_load_tfidf_array_hubble()
    elif method_name == "Voyager":
        vectorizer = storingAndLoading.dynamic_load_tfidf_vectorizer_voyager()
        X_dict = storingAndLoading.dynamic_load_tfidf_array_voyager()
    search_term_emd = vectorizer.transform([search_term])
    print(search_term_emd)
    sim_cluster = {}
    for key, term_emd in X_dict.items():
        print(term_emd)
        sim = cosine_similarity(search_term_emd, term_emd)[0][0]
        if sim > 0:
            sim_cluster["".join(key).replace("cluster_", "")] = sim
    if len(sim_cluster) > 0:
        return sim_cluster
    else:
        return "no_cluster"

    # highest_sim_cluster = ""
    # highest_sim = 0
    # for key, term_emd in X_dict.items():
    #     sim = cosine_similarity(search_term_emd, term_emd)[0][0]
    #     if sim > highest_sim:
    #         highest_sim = sim
    #         highest_sim_cluster = key
    # if highest_sim >= 0:
    #     return "".join(highest_sim_cluster).replace("cluster_", "")
    # else:
    #     return "no_cluster"


def run_WEHONA(split_entity_list_fromUI, content_depth_needed, content_capture_needed, time_place_weight,
               content_weight, topic_interest_keyword, from_date_keyword, to_date_keyword, ratio_limit):
    storingAndLoading.storeSilhoutte({})
    storingAndLoading.store_summaries_hubble({})
    cluster_info_for_not_clustered_data_dict = {}
    Nodes_dict = {}
    entity_naming_dict = {}
    content_depth_now = 1
    Place_Sentences, Person_Sentences, Content_Sentences, Day_Sentences, Month_Sentences, Year_Sentences, \
    Date_Sentences, Time_Sentences, Category_Sentences, cluster_embeddings_dict_full, docs_dict, title_dict, text_dict, ner_dict, pos_dict, weights, news_content_length, top2vec_model, category_split = storingAndLoading.loadData()
    parent_cluster_main_phase_1 = Node([x for x in range(news_content_length)])
    clusters_to_furthur_split = helper.fetchDocumentstoSplit(text_dict, Date_Sentences, topic_interest_keyword,
                                                             from_date_keyword, to_date_keyword,
                                                             news_content_length)

    if not clusters_to_furthur_split:
        raise Exception("Unable to find documents for the given filters")
    weights, possible_content_depth, weights_parameters_list = helper.create_content_weights(
        len(clusters_to_furthur_split), weights,
        content_capture_needed)

    content_weight_temp = 0.4
    pos_weight_temp = 0.6

    cluster_selection_epsilon_temp = 0.1

    weights["content_pos"] = {"content_weight": content_weight_temp, "pos_weight": pos_weight_temp}

    for idx in range(possible_content_depth):
        weights["content"][str(idx + 1)][1]["cluster_selection_epsilon"] = cluster_selection_epsilon_temp

    if content_depth_needed > possible_content_depth:  # set max value in ui
        content_depth_needed = possible_content_depth

    split_entity_list = helper.getSplitEntityList(split_entity_list_fromUI, content_depth_needed)
    print("splitting data and generating nodes and edges...")

    split_with_size = weights["content"][str(content_depth_now)][1]["min_cluster_size"]

    parent_cluster_main_phase_1, cluster_info_for_not_clustered_data_dict, clusters_to_furthur_split, Nodes_dict, \
    ids_based_on_labels, entity_name_list, entity_naming_dict, content_depth_now, splitted = splitting.split_for_3_levels(
        cluster_embeddings_dict_full, split_entity_list[0], weights, parent_cluster_main_phase_1,
        clusters_to_furthur_split, cluster_info_for_not_clustered_data_dict, Nodes_dict, entity_naming_dict, True,
        content_depth_now, time_place_weight, content_weight, category_split)

    # not using pos weighting
    if splitted:
        content_weight_temp = content_weight_temp + 0.1
        pos_weight_temp = pos_weight_temp - 0.1
        cluster_selection_epsilon_temp = cluster_selection_epsilon_temp - 0.1
        if cluster_selection_epsilon_temp < 0:
            cluster_selection_epsilon_temp = 0
        if content_weight_temp > 1:
            content_weight_temp = 1
        if pos_weight_temp < 0:
            pos_weight_temp = 0
        weights["content_pos"] = {"content_weight": content_weight_temp, "pos_weight": pos_weight_temp}
        for idx in range(possible_content_depth):
            weights["content"][str(idx + 1)][1]["cluster_selection_epsilon"] = cluster_selection_epsilon_temp

    nodes_edges_main, child_count = helper.create_nodes_edges_from_hierarchy(parent_cluster_main_phase_1, 0, 1, 0,
                                                                             entity_name_list, entity_naming_dict,
                                                                             split_with_size)
    for entity_name in split_entity_list[1:]:
        nodes_edges_main, child_count, clusters_to_furthur_split, Nodes_dict, content_depth_now, splitted = splitting.perform_furthur_split_by_entity(
            cluster_embeddings_dict_full, weights, entity_name, clusters_to_furthur_split, nodes_edges_main,
            Nodes_dict,
            child_count, cluster_info_for_not_clustered_data_dict, entity_naming_dict, content_depth_now,
            time_place_weight, content_weight)

        if splitted:
            content_weight_temp = content_weight_temp + 0.1
            pos_weight_temp = pos_weight_temp - 0.1
            cluster_selection_epsilon_temp = cluster_selection_epsilon_temp - 0.1
            if cluster_selection_epsilon_temp < 0:
                cluster_selection_epsilon_temp = 0
            if content_weight_temp > 1:
                content_weight_temp = 1
            if pos_weight_temp < 0:
                pos_weight_temp = 0
            weights["content_pos"] = {"content_weight": content_weight_temp, "pos_weight": pos_weight_temp}
            for idx in range(possible_content_depth):
                weights["content"][str(idx + 1)][1]["cluster_selection_epsilon"] = cluster_selection_epsilon_temp

    storingAndLoading.storeUseFlat({"useFlat": True})
    time_dict = {v: k for v, k in enumerate(Time_Sentences)}
    category_dict = {v: k for v, k in enumerate(Category_Sentences)}
    nodes_edges_main['docs_dict'], nodes_edges_main['text_dict'], nodes_edges_main['time_dict'], nodes_edges_main[
        'category_dict'], nodes_edges_main['pos_dict'] = docs_dict, text_dict, time_dict, category_dict, pos_dict

    # possible_content_depth_nodes_edges = 0
    # for node in nodes_edges_main['nodes']:
    #     if node["level"] > possible_content_depth_nodes_edges:
    #         possible_content_depth_nodes_edges = node["level"]

    # nodes_edges_main['possible_content_depth'] = possible_content_depth_nodes_edges

    nodes_edges_main['possible_content_depth'] = possible_content_depth
    nodes_edges_main['weights_list'] = weights_parameters_list

    # used to calculated silhoutte scores of only child events
    # nodes_edges_main = helper.remove_one_one_nodes(nodes_edges_main)  # delete once done
    # weighted_embeddings_temp = embeddings.get_weighted_embeddings(cluster_embeddings_dict_full, [0, 0, 0, 0, 1, 0, 0, 0, 1])
    # silhoutte_score_child_labels = helper.get_silhoutte_score_child_labels(nodes_edges_main)
    # return silhouette_score(weighted_embeddings_temp, silhoutte_score_child_labels)

    nodes_edges_main = filtering.eventRepresentation(nodes_edges_main, title_dict, text_dict, Place_Sentences,
                                                     Person_Sentences,
                                                     Date_Sentences, ratio_limit)
    nodes_edges_main, cluster_name_dict = filtering.filter_nodes_edges(nodes_edges_main, ner_dict, pos_dict,
                                                                       ratio_limit)
    nodes_edges_main = helper.create_cluster_match(nodes_edges_main, cluster_embeddings_dict_full)

    # nodes_edges_main = helper.find_related_events(nodes_edges_main, cluster_embeddings_dict_full, False)

    nodes_edges_main = helper.remove_one_one_nodes(nodes_edges_main)

    vectorizer, X = helper.search_tfidf(nodes_edges_main)

    # vectorizer, X = [], []

    # nodes_edges_main = helper.post_process(nodes_edges_main)

    storingAndLoading.dynamic_store_cluster_name_dict_news(cluster_name_dict)
    storingAndLoading.static_store_cluster_name_dict_news(cluster_name_dict)
    storingAndLoading.storeDynamicNews(nodes_edges_main)
    storingAndLoading.storeStaticNews(nodes_edges_main)
    storingAndLoading.dynamic_store_tfidf_vectorizer_hubble(vectorizer)
    storingAndLoading.dynamic_store_tfidf_array_hubble(X)
    storingAndLoading.static_store_tfidf_vectorizer_hubble(vectorizer)
    storingAndLoading.static_store_tfidf_array_hubble(X)


    # for calculating silhoutte score per level
    helper.calculateSilhoutte(nodes_edges_main)


def alter_WEHONA(content_depth_needed):
    static_news = storingAndLoading.load_static_news()
    static_search = storingAndLoading.static_load_cluster_name_dict_news()
    static_X = storingAndLoading.static_load_tfidf_array_hubble()
    nodes = static_news["nodes"]
    weights_list = static_news["weights_list"]
    content_depth_by_weight = static_news["weights_list"][content_depth_needed - 1]
    nodes_updated = []
    search_updated = {}
    X_updated = {}
    for node in nodes:
        if node["split_with_size"] >= content_depth_by_weight:
            nodes_updated.append(node)
            cluster_number = "cluster_" + str(node["id"])
            search_updated[cluster_number] = static_search[cluster_number]
            X_updated[cluster_number] = static_X[cluster_number]
    static_news["nodes"] = nodes_updated
    storingAndLoading.storeDynamicNews(static_news)
    storingAndLoading.dynamic_store_cluster_name_dict_news(search_updated)
    storingAndLoading.dynamic_store_tfidf_array_hubble(X_updated)


def alter_Top2Vec(content_depth_needed):
    static_top2vec_news = storingAndLoading.load_static_top2vec_news()
    static_top2vec_search = storingAndLoading.static_load_cluster_name_dict_top2vec()
    static_X = storingAndLoading.static_load_tfidf_array_voyager()
    nodes = static_top2vec_news["nodes"]
    nodes_updated = []
    search_updated = {}
    X_updated = {}
    for node in nodes:
        if node["level"] <= content_depth_needed:
            nodes_updated.append(node)
            cluster_number = "cluster_" + str(node["id"])
            search_updated[cluster_number] = static_top2vec_search[cluster_number]
            X_updated[cluster_number] = static_X[cluster_number]
    static_top2vec_news["nodes"] = nodes_updated
    storingAndLoading.storeDynamictop2vecNews(static_top2vec_news)
    storingAndLoading.dynamic_store_cluster_name_dict_top2vec(search_updated)
    storingAndLoading.dynamic_store_tfidf_array_voyager(X_updated)


# def run_nothing():
#     print("hi")


def generate_custer_summary(cluster_method_no):
    cluster_method_no_list = cluster_method_no.split(":")
    cluster_method = cluster_method_no_list[0]
    cluster_no = cluster_method_no_list[1]
    if cluster_method == "Hubble":
        summaries = storingAndLoading.load_summaries_hubble()
        if cluster_no in summaries:
            return summaries[cluster_no]
        else:
            news = storingAndLoading.load_dynamic_news()
            cluster_summary = helper.generate_custer_summary(news, cluster_no)
            summaries[cluster_no] = cluster_summary
            storingAndLoading.store_summaries_hubble(summaries)
            return cluster_summary
    elif cluster_method == "Voyager":
        summaries = storingAndLoading.load_summaries_voyager()
        if cluster_no in summaries:
            return summaries[cluster_no]
        else:
            news = storingAndLoading.load_dynamic_top2vec_news()
            cluster_summary = helper.generate_custer_summary(news, cluster_no)
            summaries[cluster_no] = cluster_summary
            storingAndLoading.store_summaries_voyager(summaries)
            return cluster_summary


def generate_custer_what(cluster_method_no):
    cluster_method_no_list = cluster_method_no.split(":")
    cluster_method = cluster_method_no_list[0]
    cluster_no = cluster_method_no_list[1]
    if cluster_method == "Hubble":
        whats = storingAndLoading.load_whats_hubble()
        if cluster_no in whats:
            return whats[cluster_no]
        else:
            news = storingAndLoading.load_dynamic_news()
            cluster_what = helper.generate_custer_what(news, cluster_no)
            whats[cluster_no] = cluster_what
            storingAndLoading.store_whats_hubble(whats)
            return cluster_what
    elif cluster_method == "Voyager":
        whats = storingAndLoading.load_whats_voyager()
        if cluster_no in whats:
            return whats[cluster_no]
        else:
            news = storingAndLoading.load_dynamic_top2vec_news()
            cluster_what = helper.generate_custer_what(news, cluster_no)
            whats[cluster_no] = cluster_what
            storingAndLoading.store_whats_voyager(whats)
            return cluster_what
