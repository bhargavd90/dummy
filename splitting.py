import hdbscan
import pandas as pd
from anytree import Node
import embeddings
import helper
from hdbscan import flat
import numpy as np
import storingAndLoading
from dbvc import DBCV
from scipy.spatial.distance import euclidean
from sklearn.metrics import silhouette_score


def get_cluster_labels(weighted_embeddings, hdbscan_dict, is_first):
    clusterer = hdbscan.HDBSCAN(**hdbscan_dict)
    clusterer.fit(weighted_embeddings)
    if 11 > len(np.unique(clusterer.labels_)) > 1:
        storingAndLoading.storeUseFlat({"useFlat": False})
    useFlat = storingAndLoading.loadUseFlat()
    if useFlat["useFlat"] and len(np.unique(clusterer.labels_)) > 10:
        storingAndLoading.storeUseFlat({"useFlat": False})
        hdbscan_dict["X"] = weighted_embeddings
        hdbscan_dict["n_clusters"] = 9
        clusterer = flat.HDBSCAN_flat(**hdbscan_dict)
        return clusterer.labels_
    else:
        return clusterer.labels_


# check what to keep for -1 label
def perform_split(cluster_data, complete_weights, hdbscan_dict, is_first):
    cluster_embeddings_dict = cluster_data[0]
    weighted_embeddings = embeddings.get_weighted_embeddings(cluster_embeddings_dict, complete_weights)
    if (len(weighted_embeddings)) > 1:
        cluster_labels = get_cluster_labels(weighted_embeddings, hdbscan_dict, is_first)

        # print_hierarchy(weighted_embeddings)
        # print("hi")
        # print(DBCV(weighted_embeddings, cluster_labels, dist_function=euclidean))
        # if len(np.unique(cluster_labels)) > 1:
        #     print(silhouette_score(weighted_embeddings, cluster_labels))

        id_mapping_dict = {v: k for v, k in enumerate(cluster_data[1])}
        list_of_lists = pd.Series(range(len(cluster_labels))).groupby(cluster_labels, sort=True).apply(list).tolist()
        correct_id_labels = [[id_mapping_dict.get(ele, ele) for ele in lst] for lst in list_of_lists]
        if -1 in cluster_labels:
            if len(correct_id_labels) == 1:
                cluster_info_for_not_clustered_data = [0]
            else:
                cluster_info_for_not_clustered_data = [y for y in range(-1, len(correct_id_labels) - 1)]
        else:
            cluster_info_for_not_clustered_data = [y for y in range(0, len(correct_id_labels))]
    else:
        correct_id_labels = [cluster_data[1]]
        cluster_info_for_not_clustered_data = [0]
    return correct_id_labels, cluster_info_for_not_clustered_data


def split_for_3_levels(cluster_embeddings_dict_full, entity_name, weights, parent_cluster_main_phase_1,
                       clusters_to_furthur_split, cluster_info_for_not_clustered_data_dict, Nodes_dict,
                       entity_naming_dict, is_first, content_depth_now, time_place_weight, content_weight):
    if entity_name == 'place':
        entity_name_list = ['place']
    elif entity_name == 'person':
        entity_name_list = ['person']
    elif entity_name == 'time':
        entity_name_list = ['year', 'month', 'day']
    elif entity_name == 'content':
        entity_name_list = ['content']

    if entity_name == "time":
        # Level_1
        parent_cluster_phase_1 = parent_cluster_main_phase_1
        complete_weights = weights[entity_name]["1"][0]
        hdbscan_dict = weights[entity_name]["1"][1]
        cluster_data = [
            embeddings.get_cluster_embeddings_dict_per_cluster(cluster_embeddings_dict_full, clusters_to_furthur_split),
            clusters_to_furthur_split, clusters_to_furthur_split]
        ids_based_on_labels, cluster_info_for_not_clustered_data = perform_split(cluster_data, complete_weights,
                                                                                 hdbscan_dict, is_first)
        cluster_info_for_not_clustered_data_dict['Level_1'] = cluster_info_for_not_clustered_data
        for cluster_temp in ids_based_on_labels:
            if cluster_temp != clusters_to_furthur_split or is_first:
                Nodes_dict[str(cluster_temp)] = Node(cluster_temp, parent=parent_cluster_phase_1)
                entity_naming_dict[str(cluster_temp)] = entity_name_list[0]
        clusters_to_furthur_split = []  # append to this in the last split

        # Level_2
        complete_weights = weights[entity_name]["2"][0]
        hdbscan_dict = weights[entity_name]["2"][1]
        cluster_info_for_not_clustered_data_list = []
        for index, cluster in enumerate(ids_based_on_labels):
            parent_cluster_phase_1 = Nodes_dict[str(cluster)]
            cluster_data_temp = [
                embeddings.get_cluster_embeddings_dict_per_cluster(cluster_embeddings_dict_full, cluster),
                cluster]
            ids_based_on_labels_temp, cluster_info_for_not_clustered_data = perform_split(cluster_data_temp,
                                                                                          complete_weights,
                                                                                          hdbscan_dict, is_first)
            cluster_info_for_not_clustered_data_list.append(cluster_info_for_not_clustered_data)
            ids_based_on_labels[index] = ids_based_on_labels_temp
            for cluster_temp in ids_based_on_labels_temp:
                if cluster_temp != cluster:
                    Nodes_dict[str(cluster_temp)] = Node(cluster_temp, parent=parent_cluster_phase_1)
                    entity_naming_dict[str(cluster_temp)] = entity_name_list[1]
        cluster_info_for_not_clustered_data_dict['Level_2'] = cluster_info_for_not_clustered_data_list

        # Level_3
        complete_weights = weights[entity_name]["3"][0]
        hdbscan_dict = weights[entity_name]["3"][1]
        cluster_info_for_not_clustered_data_list = []
        for index_1, cluster_1 in enumerate(ids_based_on_labels):
            cluster_info_for_not_clustered_data_list_1 = []
            for index_2, cluster_2 in enumerate(cluster_1):
                parent_cluster_phase_1 = Nodes_dict[str(cluster_2)]
                cluster_data_temp = [
                    embeddings.get_cluster_embeddings_dict_per_cluster(cluster_embeddings_dict_full, cluster_2),
                    cluster_2]
                ids_based_on_labels_temp, cluster_info_for_not_clustered_data = perform_split(cluster_data_temp,
                                                                                              complete_weights,
                                                                                              hdbscan_dict, is_first)
                cluster_info_for_not_clustered_data_list_1.append(cluster_info_for_not_clustered_data)
                ids_based_on_labels[index_1][index_2] = ids_based_on_labels_temp
                for cluster_temp in ids_based_on_labels_temp:
                    clusters_to_furthur_split.append(cluster_temp)
                    if cluster_temp != cluster_2:
                        Nodes_dict[str(cluster_temp)] = Node(cluster_temp, parent=parent_cluster_phase_1)
                        entity_naming_dict[str(cluster_temp)] = entity_name_list[2]
            cluster_info_for_not_clustered_data_list.append(cluster_info_for_not_clustered_data_list_1)
        cluster_info_for_not_clustered_data_dict['Level_3'] = cluster_info_for_not_clustered_data_list

    elif entity_name == "place":
        parent_cluster_phase_1 = parent_cluster_main_phase_1
        # Level_1
        complete_weights = weights[entity_name]["1"][0]
        hdbscan_dict = weights[entity_name]["1"][1]
        cluster_data = [
            embeddings.get_cluster_embeddings_dict_per_cluster(cluster_embeddings_dict_full, clusters_to_furthur_split),
            clusters_to_furthur_split, clusters_to_furthur_split]
        ids_based_on_labels, cluster_info_for_not_clustered_data = perform_split(cluster_data, complete_weights,
                                                                                 hdbscan_dict, is_first)
        cluster_info_for_not_clustered_data_dict['Level_1'] = cluster_info_for_not_clustered_data
        for cluster_temp in ids_based_on_labels:
            if cluster_temp != clusters_to_furthur_split or is_first:
                Nodes_dict[str(cluster_temp)] = Node(cluster_temp, parent=parent_cluster_phase_1)
                entity_naming_dict[str(cluster_temp)] = entity_name_list[0]
        clusters_to_furthur_split = []
        for cluster_temp in ids_based_on_labels:
            clusters_to_furthur_split.append(cluster_temp)

    elif entity_name == "person":
        parent_cluster_phase_1 = parent_cluster_main_phase_1
        # Level_1
        complete_weights = weights[entity_name]["1"][0]
        hdbscan_dict = weights[entity_name]["1"][1]
        cluster_data = [
            embeddings.get_cluster_embeddings_dict_per_cluster(cluster_embeddings_dict_full, clusters_to_furthur_split),
            clusters_to_furthur_split, clusters_to_furthur_split]
        ids_based_on_labels, cluster_info_for_not_clustered_data = perform_split(cluster_data, complete_weights,
                                                                                 hdbscan_dict, is_first)
        cluster_info_for_not_clustered_data_dict['Level_1'] = cluster_info_for_not_clustered_data
        for cluster_temp in ids_based_on_labels:
            if cluster_temp != clusters_to_furthur_split or is_first:
                Nodes_dict[str(cluster_temp)] = Node(cluster_temp, parent=parent_cluster_phase_1)
                entity_naming_dict[str(cluster_temp)] = entity_name_list[0]
        clusters_to_furthur_split = []
        for cluster_temp in ids_based_on_labels:
            clusters_to_furthur_split.append(cluster_temp)

    elif entity_name == "content":
        parent_cluster_phase_1 = parent_cluster_main_phase_1
        # Level_1
        complete_weights = weights[entity_name][str(content_depth_now)][0]
        complete_weights[5] = time_place_weight
        complete_weights[7] = time_place_weight
        complete_weights[8] = content_weight
        hdbscan_dict = weights[entity_name][str(content_depth_now)][1]
        cluster_data = [
            embeddings.get_cluster_embeddings_dict_per_cluster(cluster_embeddings_dict_full, clusters_to_furthur_split),
            clusters_to_furthur_split, clusters_to_furthur_split]
        ids_based_on_labels, cluster_info_for_not_clustered_data = perform_split(cluster_data, complete_weights,
                                                                                 hdbscan_dict, is_first)
        cluster_info_for_not_clustered_data_dict['Level_1'] = cluster_info_for_not_clustered_data
        for cluster_temp in ids_based_on_labels:
            if cluster_temp != clusters_to_furthur_split or is_first:
                Nodes_dict[str(cluster_temp)] = Node(cluster_temp, parent=parent_cluster_phase_1)
                entity_naming_dict[str(cluster_temp)] = entity_name_list[0]
        clusters_to_furthur_split = []
        for cluster_temp in ids_based_on_labels:
            clusters_to_furthur_split.append(cluster_temp)

        if is_first:
            content_depth_now = content_depth_now + 1

    return parent_cluster_main_phase_1, cluster_info_for_not_clustered_data_dict, clusters_to_furthur_split, Nodes_dict, ids_based_on_labels, entity_name_list, entity_naming_dict, content_depth_now


def perform_furthur_split_by_entity(cluster_embeddings_dict_full, weights, entity_name, clusters_to_furthur_split,
                                    nodes_edges_main, Nodes_dict, child_count, cluster_info_for_not_clustered_data_dict,
                                    entity_naming_dict, content_depth_now, time_place_weight, content_weight):
    clusters_to_furthur_split_whole = []
    for clus_to_spl in clusters_to_furthur_split:
        for key, value in nodes_edges_main['cluster_dict'].items():
            if value == clus_to_spl:
                parent_count = int(key.split('_')[1]) + 1
                break
        parent_cluster_node = Nodes_dict[str(clus_to_spl)]
        parent_cluster_main_phase_1, cluster_info_for_not_clustered_data_dict, clusters_to_furthur_split, Nodes_dict, ids_based_on_labels, entity_name_list, entity_naming_dict, content_depth_now = split_for_3_levels(
            cluster_embeddings_dict_full, entity_name, weights, parent_cluster_node, clus_to_spl,
            cluster_info_for_not_clustered_data_dict, Nodes_dict, entity_naming_dict, False, content_depth_now,
            time_place_weight, content_weight)
        nodes_edges_sub, child_count_sub = helper.create_nodes_edges_from_hierarchy(parent_cluster_main_phase_1,
                                                                                    parent_count,
                                                                                    child_count,
                                                                                    parent_cluster_node.depth,
                                                                                    entity_name_list,
                                                                                    entity_naming_dict)
        child_count = child_count_sub
        nodes_edges_main['nodes'] = nodes_edges_main['nodes'] + nodes_edges_sub['nodes']
        nodes_edges_main['edges'] = nodes_edges_main['edges'] + nodes_edges_sub['edges']
        nodes_edges_main['cluster_dict'] = {**nodes_edges_main['cluster_dict'], **nodes_edges_sub['cluster_dict']}
        for clus in clusters_to_furthur_split:
            clusters_to_furthur_split_whole.append(clus)
    if entity_name == "content":
        content_depth_now = content_depth_now + 1
    return nodes_edges_main, child_count, clusters_to_furthur_split_whole, Nodes_dict, content_depth_now
