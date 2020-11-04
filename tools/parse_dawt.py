import re
import json
from transformers import FlaubertTokenizer
from tqdm import tqdm

"""
    Read and transform DAWT file to a new format handled by NER
"""

def parse_dawt(label_map, max_seq_length=64, pad_token_label_id=0):
    sample = []
    labels = []
    label_counts = {key: 0 for key, _ in label_map.items()}

    annotations = open("../../opendata/wiki_annotation/wiki_annotations_json_sample_fr")
    tokenizer = FlaubertTokenizer.from_pretrained("jplu/tf-flaubert-base-cased")

    for annotation in tqdm(annotations):
        a = json.loads(annotation)

        if "entities" not in a or "tokens" not in a:
            continue

        entities = a["entities"]
        tokens = a["tokens"]
        del a

        # Add entities to tokens
        for entity in entities:
            i = entity["start_position"]
            token = tokens[i]

            if "type" not in entity:
                continue

            entity.pop("id_str", None)
            if "entity" not in token and (entity["end_position"] - entity["start_position"] == 0):
                token["entity"] = entity
            i += 1

        word_tokens = []
        label_ids = []

        for idx, token in enumerate(tokens):
            word = token["raw_form"]
            label = token["entity"]["type"] if "entity" in token else 'O'

            if idx != len(tokens) and word.lower() in ["l", "d", "s", "t", "n", "j", "m", "n"]:
                word += "\'" 
                tokens[idx]["raw_form"] = word 

            word_token = tokenizer.tokenize(word)
            word_tokens.extend(word_token)
            label_ids.extend([label_map[label]] + [0] * (len(word_token) - 1))
            label_counts[label] += 1 * len(word_token)
            
            if "section_break" in token:
                word_tokens = [tokenizer.cls_token] + word_tokens + [tokenizer.sep_token]

                padding_length = max_seq_length - len(word_tokens)

                label_ids = [pad_token_label_id] + label_ids + [pad_token_label_id]
                input_ids = tokenizer.convert_tokens_to_ids(word_tokens + [tokenizer.pad_token] * padding_length)
                label_ids += [pad_token_label_id] * padding_length
                sample.append(input_ids[:max_seq_length])
                labels.append(label_ids[:max_seq_length])
                word_tokens = []
                label_ids = []

    return sample, labels, label_counts

# if __name__ == "__main__":
#     s, l = parse_dawt()
#     print(s[0])
#     print(l[0])