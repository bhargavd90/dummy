# def get_country_continent(place_name):
#     place_name = re.sub('[^A-Za-z0-9 ]+', '', place_name)
#     try:
#         countries.get(place_name)
#         country = place_name.capitalize()
#     except:
#         country = str(geolocator.geocode(place_name)).split(",")[-1].split("/")[-1].strip()
#
#     if (place_name.lower() == country.lower()):
#         city = "city unknown"
#     else:
#         city = place_name
#     if country == "Deutschland":
#         country = "Germany"
#     elif country == "Us":
#         country = "United States"
#     elif country == 'Belgien':
#         country = "Belgium"
#     try:
#         country_code = pc.country_name_to_country_alpha2(country, cn_name_format="default")
#         continent_code = pc.country_alpha2_to_continent_code(country_code)
#         continent_name = pc.convert_continent_code_to_continent_name(continent_code)
#         return (continent_name, country, city)
#     except:
#         return ("continent unknown", country, city)
#
#
# def place_sentences_to_country_continent_city(Place_Sentences):
#     continent_list = []
#     country_list = []
#     city_list = []
#     for news_place_sentence in Place_Sentences:
#         continent_country_city = []
#         news_place_list = news_place_sentence.split(',')
#         for news_place in news_place_list:
#             (continent, country, city) = get_country_continent(news_place)
#             if country != 'None':
#                 continent_country_city.append(get_country_continent(news_place))
#         ctr = Counter(continent_country_city)
#         most_common = ctr.most_common(2)
#         if len(most_common) > 1:
#             continent_country_city_max_1 = most_common[0]
#             continent_country_city_max_2 = most_common[1]
#             if not all(continent_country_city_max_1[0]) and all(continent_country_city_max_2[0]):
#                 continent_country_city_max = continent_country_city_max_2
#             else:
#                 continent_country_city_max = continent_country_city_max_1
#         elif len(most_common) == 1:
#             continent_country_city_max = most_common[0]
#         else:
#             continent_country_city_max = (('continent unknown', 'country unknown', 'city unknown'), 0)
#         continent_list.append(continent_country_city_max[0][0])
#         country_list.append(continent_country_city_max[0][1])
#         city_list.append(continent_country_city_max[0][2])
#     return continent_list, country_list, city_list
#
#
# def print_hierarchy(weighted_embeddings):
#     Z = hierarchy.linkage(weighted_embeddings, 'ward')
#     plt.figure()
#     figure(figsize=(15, 8), dpi=80)
#     dn = hierarchy.dendrogram(Z, color_threshold=3)
#
#
# # for building a tree if we are using Top2Vec
# def find_parent(topic_child_list, previous_topic_child_list, parent_cluster_main, child_dict):
#     if topic_child_list == previous_topic_child_list:
#         return parent_cluster_main
#     else:
#         for current_child in topic_child_list:
#             search_child = current_child[0]
#             for previous_child in previous_topic_child_list:
#                 if search_child in previous_child:
#                     if len(current_child) == len(previous_child):
#                         continue
#                     else:
#                         return child_dict[str(previous_child)]
#
#
# def create_hierarchical_tree_from_cluster_docs(model):
#     check_list = []
#     child_dict = {}
#     keywords_dict = {}
#     topic_vector_dict = {}
#     sim_dict = {}
#     total_topics = model.get_num_topics()
#     first_cluster_list = model.hierarchical_topic_reduction(1)
#     parent_cluster_main = Node(first_cluster_list[0])
#     topic_words, word_scores, topic_nums = model.get_topics(reduced=True)
#     topic_vector = model.topic_vectors_reduced
#     keywords_dict[str(first_cluster_list[0])] = topic_words
#     topic_vector_dict[str(first_cluster_list[0])] = topic_vector
#     # chceck_list.append(model.hierarchical_topic_reduction(1))
#     previous_topic_child_list = model.hierarchical_topic_reduction(2)
#     for i in range(total_topics - 2):
#         topic_child_list = model.hierarchical_topic_reduction(i + 2)
#         topic_words, word_scores, topic_nums = model.get_topics(reduced=True)
#         topic_vectors = model.topic_vectors_reduced
#         parent_cluster = find_parent(topic_child_list, previous_topic_child_list, parent_cluster_main, child_dict)
#         sim_name = ""
#         sim_vectors = []
#         for topic_child, topic_word, topic_vector in zip(topic_child_list, topic_words, topic_vectors):
#             if topic_child not in check_list:
#                 sim_name = sim_name + str(topic_child) + "_"  # str(topic_child[0:2])+"_"
#                 sim_vectors.append(topic_vector)
#                 keywords_dict[str(topic_child)] = topic_word
#                 topic_vector_dict[str(topic_child)] = topic_vector
#                 child_dict[str(topic_child)] = Node(topic_child, parent=parent_cluster)
#                 check_list.append(topic_child)
#         sim_dict[sim_name.strip("_")] = round(cosine_similarity([sim_vectors[0]], [sim_vectors[1]]).item(), 3)
#         previous_topic_child_list = topic_child_list
#     return parent_cluster_main, keywords_dict, topic_vector_dict, sim_dict


# def perform_last_levels_split(cluster_embeddings_dict_full, weights, entity_name, clusters_to_furthur_split,
# nodes_edges_main, Nodes_dict, child_count): complete_weights = weights[entity_name]["1"][0] hdbscan_dict = weights[
# entity_name]["1"][1] for clus_to_spl in clusters_to_furthur_split: cluster_ids_array = np.array(clus_to_spl) for key,
# value in nodes_edges_main['cluster_dict'].items(): if value == clus_to_spl: parent_count = int(key.split('_')[1])+1
# break parent_cluster_node = Nodes_dict[str(clus_to_spl)]
#
# # Using top2vec for hierarchical clusters # cluster_docs = [news_content[clus_id] for clus_id in cluster_ids_array]
# model = Top2Vec(cluster_docs) # parent_cluster_main_phase_2, keywords_dict, topic_vector_dict, sim_dict =
# create_hierarchical_tree_from_cluster_docs(model) # docs_dict_sub = get_docs_from_topic(cluster_ids_array,
# news_publisher_title, model)
#
# # Using HDBSCAN for getting n clusters and appending them to parent cluster. cluster_data_temp = [
# get_cluster_embeddings_dict_per_cluster(cluster_embeddings_dict_full, cluster_ids_array), cluster_ids_array]
# ids_based_on_labels_temp, cluster_info_for_not_clustered_data = perform_split(cluster_data_temp, complete_weights,
# hdbscan_dict) parent_cluster_main_phase_2 = create_hierarchical_tree_from_cluster_docs_testing_without_hierarchy(
# ids_based_on_labels_temp, parent_cluster_node) nodes_edges_sub, child_count_sub =
# create_nodes_edges_from_hierarchy(parent_cluster_main_phase_2, parent_count, child_count,
# parent_cluster_node.depth, ['content'], 'every_node_content') # only one content because of only one solit at last
# child_count = child_count_sub
#
#     nodes_edges_main['nodes'] = nodes_edges_main['nodes'] + nodes_edges_sub['nodes']
#     nodes_edges_main['edges'] = nodes_edges_main['edges'] + nodes_edges_sub['edges']
#     nodes_edges_main['cluster_dict'] = {**nodes_edges_main['cluster_dict'], **nodes_edges_sub['cluster_dict']}
#
#   return nodes_edges_main


# to remove one on one edges in "filter_nodes_edges" method
# edges_dict_updated_temp = edges_dict_updated
# for edgs_fltr in edges_filter_list_2:
#   for edge in edges_dict_updated_temp:
#     if edge['to'] == edgs_fltr:
#       cluster_no_from = edge['from']
#       cluster_label_1 = edge['label']
#     if edge['from'] == edgs_fltr:
#       cluster_no_to = edge['to']
#       cluster_label_2 = edge['label']
#   new_edge = {'from': cluster_no_from, 'to': cluster_no_to, 'label':cluster_label_1, 'font': {'align': 'middle'}}
#   del_edge_1 = {'from': cluster_no_from, 'to': edgs_fltr, 'label':cluster_label_1, 'font': {'align': 'middle'}}
#   del_edge_2 = {'from': edgs_fltr, 'to': cluster_no_to, 'label':cluster_label_2, 'font': {'align': 'middle'}}
#   print(del_edge_2)
#   edges_dict_updated.remove(del_edge_1)
#   edges_dict_updated.remove(del_edge_2)
#   edges_dict_updated.append(new_edge)
