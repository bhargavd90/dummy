from sentence_transformers import SentenceTransformer
import umap as umap
import numpy as np

#from gensim.models import Word2Vec
# model = Word2Vec(sentences=sentences_list, vector_size=100, window=5, min_count=1, workers=4)
# model.save("word2vec.model")


sbert_model_name = 'paraphrase-MiniLM-L3-v2'
sbert_model = SentenceTransformer(sbert_model_name)


def get_embeddings(sentences_list, content_flag, unique_ner_dict, unique_pos_dict):
    if not content_flag:
        embeddings_list = sbert_model.encode(sentences_list)
    else:

        # embeddings_list = []
        # for index, news_article in enumerate(sentences_list):
        #     pos_embeddings_list = []
        #     ner_embeddings_list = []
        #     news_article_embedding = sbert_model.encode(news_article)
        #     pos_list = unique_pos_dict[index]
        #     ner_list = unique_ner_dict[index]
        #     for pos_word in pos_list:
        #         pos_embeddings_list.append(pos_word[1] * sbert_model.encode(pos_word[0]))
        #     for ner_word in ner_list:
        #         ner_embeddings_list.append(pos_word[1] * sbert_model.encode(ner_word[0]))
        #     pos_embedding = np.mean(pos_embeddings_list, axis=0)
        #     ner_embedding = np.mean(ner_embeddings_list, axis=0)
        #     embeddings_list.append([news_article_embedding, pos_embedding, ner_embedding])

        # embeddings_list = []
        # for index, news_article in enumerate(sentences_list):
        #     pos_embeddings_list = []
        #     ner_embeddings_list = []
        #     news_article_embedding = sbert_model.encode(news_article)
        #     pos_list = unique_pos_dict[index]
        #     ner_list = unique_ner_dict[index]
        #     for pos_word in pos_list:
        #         pos_embeddings_list.append(pos_word[1] * sbert_model.encode(pos_word[0]))
        #     for ner_word in ner_list:
        #         ner_embeddings_list.append(pos_word[1] * sbert_model.encode(ner_word[0]))
        #     pos_embedding = np.mean(pos_embeddings_list, axis=0)
        #     ner_embedding = np.mean(ner_embeddings_list, axis=0)
        #     embeddings_list.append((0.4 * news_article_embedding) + (0.6 * pos_embedding) + (0 * ner_embedding))

        # umap_dict = {'n_neighbors': 30, 'min_dist': 0.0, 'n_components': 5, 'random_state': 42}
        # embeddings_list = umap.UMAP(**umap_dict).fit_transform(sbert_model.encode(sentences_list))

        embeddings_list = sbert_model.encode(sentences_list)


    return embeddings_list


def get_cluster_embeddings_full(model_name, Place_Sentences, Person_Sentences, Content_Sentences, Day_Sentences,
                                Month_Sentences, Year_Sentences, title, umap_flag, umap_dict, unique_ner_dict,
                                unique_pos_dict):
    print("generating embeddings for data ...")
    if umap_flag:
        cluster_embeddings_dict = {'place_embeddings': umap.UMAP(**umap_dict).fit_transform(
            get_embeddings(Place_Sentences, False, unique_ner_dict, unique_pos_dict)),
            'person_embeddings': umap.UMAP(**umap_dict).fit_transform(
                get_embeddings(Person_Sentences, False, unique_ner_dict, unique_pos_dict)),
            'content_embeddings': get_embeddings(Content_Sentences, True, unique_ner_dict, unique_pos_dict),
            'day_embeddings': umap.UMAP(**umap_dict).fit_transform(
                get_embeddings(Day_Sentences, False, unique_ner_dict, unique_pos_dict)),
            'month_embeddings': umap.UMAP(**umap_dict).fit_transform(
                get_embeddings(Month_Sentences, False, unique_ner_dict, unique_pos_dict)),
            'year_embeddings': umap.UMAP(**umap_dict).fit_transform(
                get_embeddings(Year_Sentences, False, unique_ner_dict, unique_pos_dict)),
            "title_embeddings": umap.UMAP(**umap_dict).fit_transform(
                get_embeddings(title, False, unique_ner_dict, unique_pos_dict))
        }
    else:
        cluster_embeddings_dict = {
            'place_embeddings': get_embeddings(Place_Sentences, False, unique_ner_dict, unique_pos_dict),
            'person_embeddings': get_embeddings(Person_Sentences, False, unique_ner_dict, unique_pos_dict),
            'content_embeddings': get_embeddings(Content_Sentences, True, unique_ner_dict, unique_pos_dict),
            'day_embeddings': get_embeddings(Day_Sentences, False, unique_ner_dict, unique_pos_dict),
            'month_embeddings': get_embeddings(Month_Sentences, False, unique_ner_dict, unique_pos_dict),
            'year_embeddings': get_embeddings(Year_Sentences, False, unique_ner_dict, unique_pos_dict),
            "title_embeddings": get_embeddings(title, False, unique_ner_dict, unique_pos_dict)}
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


def get_cluster_embeddings_dict_per_cluster(cluster_embeddings_dict_full, cluster_data, content_pos_weight={}):
    cluster_embeddings_dict_per_cluster = {}
    for key, value in cluster_embeddings_dict_full.items():
        if key == "content_embeddings":

            # content_embeddings_per_cluster = []
            # for id in cluster_data:
            #     embeddings_list = value[id]
            #     weighted_embeddings_content_pos_ner = (content_pos_weight["content_weight"] * embeddings_list[0]) + (content_pos_weight["pos_weight"] * embeddings_list[1]) + (0 *embeddings_list[2])
            #     content_embeddings_per_cluster.append(weighted_embeddings_content_pos_ner)
            # umap_dict = {'n_neighbors': 30, 'min_dist': 0.0, 'n_components': 5, 'random_state': 42}
            # cluster_embeddings_dict_per_cluster[key] = umap.UMAP(**umap_dict).fit_transform(content_embeddings_per_cluster)

            cluster_embeddings_dict_per_cluster[key] = value[[cluster_data]]

        else:
            cluster_embeddings_dict_per_cluster[key] = value[[cluster_data]]
    return cluster_embeddings_dict_per_cluster
