"""
    Read file with JSON format and prepare for POS-Tagging
"""
import re
import csv
import json
from tqdm import tqdm

out_file = open("./test.txt", "w")
conllu_file = open("../dataset/custom_dataset/aws_entities.json")
types = set([])
for idx, row in tqdm(enumerate(conllu_file)):
    article = json.loads(row)
    
    if "entities" not in article or "titre" not in article:
        continue

    for idx_entity, entity in enumerate(article["entities"]):
        types.add(entity["type"])

        words = entity["Text"].split(" ")
        if len(words) > 1:
            article["entities"][idx_entity]["Text"] = words[0]
            for word in words[1:]:
                article["entities"].append({"Text": word, "type": entity["type"]})
            
    #print("#", [entity["Text"] for entity in article["entities"]], article["titre"])
    for word in re.split(r"(\W|\b)", article["titre"]):
        word = word.strip()
        if word == "":
            continue
        is_entity = False
        for entity in article["entities"]:
            if word == entity["Text"]:
                out_file.write(word + " " + entity["type"] + "\n")
                is_entity = True
                break
        if not is_entity:
            out_file.write(word + " O\n")
    out_file.write("\n")
print(list(types))