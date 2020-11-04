import bson
import numpy as np
import pymongo
from tqdm import tqdm
from transformers import FlaubertTokenizer

from utils import post_token_classification
from flaubert_token_classification import TFFlaubertForTokenClassification

model = TFFlaubertForTokenClassification.from_pretrained("../models/ner")
tokenizer = FlaubertTokenizer.from_pretrained("jplu/tf-flaubert-base-cased")
SEQUENCE_LENGTH = 64
ner_labels = ["LOC", "MISC", "ORG", "PER", "O"]

client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["bert_clustering"]

articles = db["articles"].find()

for article in tqdm(articles):
    sentence = tokenizer.tokenize(article["raw"])
    input_tokens = tokenizer.encode_plus(
        article["raw"],
        max_length=SEQUENCE_LENGTH,
        pad_to_max_length=SEQUENCE_LENGTH,
        add_special_tokens=True,
        return_tensors='tf',
        return_token_type_ids=True,
        return_attention_mask=True,
    )

    inputs = {
        "attention_mask": input_tokens["attention_mask"],
        "token_type_ids": input_tokens["token_type_ids"],
        "training": False
    }

    # Get entities
    ner_output = model(input_tokens["input_ids"], **inputs)[0]
    ner_output = np.argmax(ner_output, axis=2)[0]
    entities = post_token_classification(sentence, ner_output, ner_labels)

    db["articles"].find_one_and_update(
        {"_id": article["_id"]},
        {"$set": {"entities": entities} }
    )
