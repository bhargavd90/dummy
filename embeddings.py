from sentence_transformers import SentenceTransformer
import umap as umap
import numpy as np

# from gensim.models import Word2Vec
# model = Word2Vec(sentences=sentences_list, vector_size=100, window=5, min_count=1, workers=4)
# model.save("word2vec.model")


sbert_model_name = 'paraphrase-MiniLM-L3-v2'
sbert_model = SentenceTransformer(sbert_model_name)


def get_embeddings(sentences_list, content_flag, unique_ner_list, unique_pos_list, unique_token_list_full):
    if content_flag == False:
        embeddings_list = sbert_model.encode(sentences_list)
    else:
        token_dict_full = {}
        embeddings_list = []
        for token in unique_token_list_full:
            token_dict_full[token] = sbert_model.encode(token)
        for _, value in content_flag.items():
            sent_list = []
            for token in value:
                if token in unique_ner_list:
                    is_ner = True
                else:
                    is_ner = False
                if token in unique_pos_list:
                    is_pos = True
                else:
                    is_pos = False
                sent_list.append([token_dict_full[token], is_ner, is_pos])
            embeddings_list.append(sent_list)
    return embeddings_list


def get_cluster_embeddings_full(model_name, Place_Sentences, Person_Sentences, Content_Sentences, Day_Sentences,
                                Month_Sentences, Year_Sentences, title, umap_flag, umap_dict, token_dict,
                                unique_ner_list, unique_pos_list, unique_token_list_full):
    print("generating embeddings for data ...")
    if umap_flag:
        cluster_embeddings_dict = {'place_embeddings': umap.UMAP(**umap_dict).fit_transform(
            get_embeddings(Place_Sentences, False, unique_ner_list, unique_pos_list, unique_token_list_full)),
            'person_embeddings': umap.UMAP(**umap_dict).fit_transform(
                get_embeddings(Person_Sentences, False, unique_ner_list, unique_pos_list, unique_token_list_full)),
            'content_embeddings': get_embeddings(Content_Sentences, token_dict, unique_ner_list,
                                                 unique_pos_list, unique_token_list_full),
            'day_embeddings': umap.UMAP(**umap_dict).fit_transform(
                get_embeddings(Day_Sentences, False, unique_ner_list, unique_pos_list, unique_token_list_full)),
            'month_embeddings': umap.UMAP(**umap_dict).fit_transform(
                get_embeddings(Month_Sentences, False, unique_ner_list, unique_pos_list, unique_token_list_full)),
            'year_embeddings': umap.UMAP(**umap_dict).fit_transform(
                get_embeddings(Year_Sentences, False, unique_ner_list, unique_pos_list, unique_token_list_full)),
            "title_embeddings": umap.UMAP(**umap_dict).fit_transform(
                get_embeddings(title, False, unique_ner_list, unique_pos_list, unique_token_list_full))
        }
    else:
        cluster_embeddings_dict = {
            'place_embeddings': get_embeddings(Place_Sentences, False, unique_ner_list, unique_pos_list,
                                               unique_token_list_full),
            'person_embeddings': get_embeddings(Person_Sentences, False, unique_ner_list, unique_pos_list,
                                                unique_token_list_full),
            'content_embeddings': get_embeddings(Content_Sentences, token_dict, unique_ner_list,
                                                 unique_pos_list, unique_token_list_full),
            'day_embeddings': get_embeddings(Day_Sentences, False, unique_ner_list, unique_pos_list,
                                             unique_token_list_full),
            'month_embeddings': get_embeddings(Month_Sentences, False, unique_ner_list, unique_pos_list,
                                               unique_token_list_full),
            'year_embeddings': get_embeddings(Year_Sentences, False, unique_ner_list, unique_pos_list,
                                              unique_token_list_full),
            "title_embeddings": get_embeddings(title, False, unique_ner_list, unique_pos_list, unique_token_list_full)}
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
        if key == "content_embeddings":
            content_embeddings_per_cluster = []
            for id in cluster_data:
                article_embeddings = []
                article_tokens = value[id]
                for token in article_tokens:
                    emb = token[0]
                    is_ner = token[1]
                    is_pos = token[2]
                    article_embeddings.append(emb)
                article_embeddings_average = np.mean(article_embeddings, axis=0)
                content_embeddings_per_cluster.append(article_embeddings_average)
            cluster_embeddings_dict_per_cluster[key] = content_embeddings_per_cluster
        else:
            cluster_embeddings_dict_per_cluster[key] = value[[cluster_data]]
    return cluster_embeddings_dict_per_cluster
