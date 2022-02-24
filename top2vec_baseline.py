from anytree import Node
import storingAndLoading
import helper
import filtering


def find_parent(topic_child_list, previous_topic_child_list, parent_cluster_main, child_dict):
    if topic_child_list == previous_topic_child_list:
        return parent_cluster_main
    else:
        for current_child in topic_child_list:
            search_child = current_child[0]
            for previous_child in previous_topic_child_list:
                if search_child in previous_child:
                    if len(current_child) == len(previous_child):
                        continue
                    else:
                        return child_dict[str(previous_child)]


def create_hierarchical_tree_from_cluster_docs(model):
    check_list = []
    child_dict = {}
    total_topics = model.get_num_topics()
    first_cluster_list = model.hierarchical_topic_reduction(1)
    parent_cluster_main = Node(first_cluster_list[0])
    previous_topic_child_list = model.hierarchical_topic_reduction(2)
    for i in range(total_topics - 2):
        topic_child_list = model.hierarchical_topic_reduction(i + 2)
        parent_cluster = find_parent(topic_child_list, previous_topic_child_list, parent_cluster_main, child_dict)
        for topic_child in topic_child_list:
            if topic_child not in check_list:
                child_dict[str(topic_child)] = Node(topic_child, parent=parent_cluster)
                check_list.append(topic_child)
        previous_topic_child_list = topic_child_list
    return parent_cluster_main


def run_Top2Vec():
    ratio_limit = 95
    Place_Sentences, Person_Sentences, Content_Sentences, Day_Sentences, Month_Sentences, Year_Sentences, \
    Date_Sentences, cluster_embeddings_dict_full, docs_dict, title_dict, text_dict, ner_dict, pos_dict, weights, news_content_length, top2vec_model = storingAndLoading.loadData()
    parent_cluster_main = create_hierarchical_tree_from_cluster_docs(top2vec_model)
    top2vec_nodes_edges_main, child_count = helper.create_nodes_edges_from_hierarchy(parent_cluster_main, 0, 1, 0,
                                                                                     "",
                                                                                     "every_node_content")
    topic_sizes, topic_nums = top2vec_model.get_topic_sizes()
    zip_iterator = zip(topic_nums, topic_sizes)
    topic_num_size_dict = dict(zip_iterator)
    cluster_dict_updated = {}
    cluster_dict = top2vec_nodes_edges_main["cluster_dict"]
    for cluster_id, topic_numbers in cluster_dict.items():
        docs_in_cluster = []
        for topic_no in topic_numbers:
            document_ids = list(
                top2vec_model.search_documents_by_topic(topic_num=topic_no, num_docs=topic_num_size_dict[topic_no])[2])
            docs_in_cluster = docs_in_cluster + document_ids
        cluster_dict_updated[cluster_id] = docs_in_cluster
    top2vec_nodes_edges_main["cluster_dict"] = cluster_dict_updated
    top2vec_nodes_edges_main['docs_dict'], top2vec_nodes_edges_main['text_dict'] = docs_dict, text_dict

    possible_content_depth_nodes_edges = 0
    for node in top2vec_nodes_edges_main['nodes']:
        if node["level"] > possible_content_depth_nodes_edges:
            possible_content_depth_nodes_edges = node["level"]

    top2vec_nodes_edges_main['possible_content_depth'] = possible_content_depth_nodes_edges

    top2vec_nodes_edges_main = filtering.eventRepresentation(top2vec_nodes_edges_main, title_dict, text_dict,
                                                             Place_Sentences,
                                                             Person_Sentences,
                                                             Date_Sentences, ratio_limit)
    top2vec_nodes_edges_main, cluster_name_dict = filtering.filter_nodes_edges(top2vec_nodes_edges_main, ner_dict,
                                                                               pos_dict,
                                                                               ratio_limit)

    top2vec_nodes_edges_main = helper.find_related_events(top2vec_nodes_edges_main, cluster_embeddings_dict_full)

    storingAndLoading.dynamic_store_cluster_name_dict_top2vec(cluster_name_dict)
    storingAndLoading.static_store_cluster_name_dict_top2vec(cluster_name_dict)
    storingAndLoading.storeDynamictop2vecNews(top2vec_nodes_edges_main)
    storingAndLoading.storeStatictop2vecNews(top2vec_nodes_edges_main)
