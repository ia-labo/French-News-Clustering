import unicodedata
import numpy as np
from datetime import datetime


def pair_list(input_list):
    result = []
    for p1 in range(len(input_list)):
        for p2 in range(p1+1,len(input_list)):
            result.append([input_list[p1], input_list[p2]])
    return result


def preprocess_title(title):
    return ''.join((c for c in unicodedata.normalize('NFD', title) if unicodedata.category(c) != 'Mn'))


def sigmoid(x):
    return 1/(1 + np.exp(-x)) 


def weighted_jaccard_similarity(sample, micro_cluster):
    y_true_set = set(list(micro_cluster.entities.keys()))
    y_pred_set = set([entity["word"] for entity in sample.entities])

    intersection = y_true_set.intersection(y_pred_set)
    union = y_true_set.union(y_pred_set)

    weighted_union = 0
    weighted_intersection = 0
    for elem in intersection:
        weighted_intersection += micro_cluster.entities[elem]
        weighted_union += micro_cluster.entities[elem]
    weighted_union += len(union) - len(intersection)

    if weighted_union == 0:
        return 0
    return weighted_intersection / weighted_union


def date_from_request(request, name, allow_missing=False):
    new_date = None
    if not request.args.get(name) and not allow_missing:
        return None, {"error": "`"+ name +"` parameter is mandatory"}
    elif not request.args.get(name):
        return datetime.now(), None

    try:
        new_date = datetime.strptime(request.args.get(name), "%Y-%m-%dT%H:%M")
    except Exception:
        return None, {"error": "`"+ name +"` has an invalid data type, must be `strptime(%Y-%m-%dT%H:%M)` compatible"}
    
    return new_date, None


def post_token_classification(sentence, ner_output, ner_labels, normalize=False):
    entities = []

    new_idx = 0
    new_sentence = [""] * (len([s for s in sentence if "</w>" in s]) + 1)
    for idx, word in enumerate(sentence):
        new_sentence[new_idx] += word.replace("</w>", "")
        if "</w>" in word:
            if ner_output[idx + 1] != 8:
                entities.append({
                    "word": new_sentence[new_idx],
                    "type": ner_labels[ner_output[idx + 1] % 4],
                    "idx": new_idx,
                })
            new_idx += 1


    tmp_idx = 0
    for entity in entities[1:]:
        if entity["idx"] - 1 == entities[tmp_idx]["idx"]:
            word = entities[tmp_idx]["word"] + " " + entity["word"]
            if normalize:
                entities[tmp_idx + 1]["word"] = str.lower(word).strip()
            else:
                entities[tmp_idx + 1]["word"] = word
            del entities[tmp_idx]
        else:
            tmp_idx += 1
    
    return entities