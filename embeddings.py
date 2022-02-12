from sentence_transformers import SentenceTransformer
import umap as umap


def get_cluster_embeddings_full(model_name, Place_Sentences, Person_Sentences, Content_Sentences, Day_Sentences,
                                Month_Sentences, Year_Sentences, title, umap_flag, umap_dict):
    print("generating embeddings for data ...")
    model = SentenceTransformer(model_name)
    if umap_flag:
        cluster_embeddings_dict = {'place_embeddings': umap.UMAP(**umap_dict).fit_transform(
            model.encode(Place_Sentences)), 'person_embeddings': umap.UMAP(**umap_dict).fit_transform(
            model.encode(Person_Sentences)), 'content_embeddings': umap.UMAP(**umap_dict).fit_transform(
            model.encode(Content_Sentences)),
            'day_embeddings': umap.UMAP(**umap_dict).fit_transform(model.encode(Day_Sentences)),
            'month_embeddings': umap.UMAP(**umap_dict).fit_transform(
                model.encode(Month_Sentences)),
            'year_embeddings': umap.UMAP(**umap_dict).fit_transform(model.encode(Year_Sentences)),
            "title_embeddings": umap.UMAP(**umap_dict).fit_transform(model.encode(title))
                                    }
    else:
        cluster_embeddings_dict = {'place_embeddings': model.encode(Place_Sentences),
                                   'person_embeddings': model.encode(Person_Sentences),
                                   'content_embeddings': model.encode(Content_Sentences),
                                   'day_embeddings': model.encode(Day_Sentences),
                                   'month_embeddings': model.encode(Month_Sentences),
                                   'year_embeddings': model.encode(Year_Sentences),
                                   "title_embeddings": model.encode(title)}
    return cluster_embeddings_dict


def get_weighted_embeddings(cluster_embeddings_dict, complete_weights):
    time_embeddings = (complete_weights[0] * cluster_embeddings_dict['day_embeddings']) + (
            complete_weights[1] * cluster_embeddings_dict['month_embeddings']) + (
                              complete_weights[2] * cluster_embeddings_dict['year_embeddings'])
    content_embeddings = (complete_weights[3] * cluster_embeddings_dict['title_embeddings']) + (
            complete_weights[4] * cluster_embeddings_dict['content_embeddings'])
    weighted_embeddings = ((complete_weights[5] * cluster_embeddings_dict['place_embeddings']) + (
            complete_weights[6] * cluster_embeddings_dict['person_embeddings']) + (
                                   complete_weights[7] * time_embeddings) + (
                                   complete_weights[8] * content_embeddings) / 4)
    return weighted_embeddings


def get_cluster_embeddings_dict_per_cluster(cluster_embeddings_dict_full, cluster_data):
    cluster_embeddings_dict_per_cluster = {}
    for key, value in cluster_embeddings_dict_full.items():
        cluster_embeddings_dict_per_cluster[key] = value[[cluster_data]]
    return cluster_embeddings_dict_per_cluster
