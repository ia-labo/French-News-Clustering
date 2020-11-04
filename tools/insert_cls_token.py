from tqdm import tqdm
import numpy as np
import bson
import pymongo
from transformers.modeling_tf_flaubert import TFFlaubertForSequenceClassification


model = TFFlaubertForSequenceClassification.from_pretrained("../models/categorisation")


client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["bert_clustering"]

articles = db["articles"].find()

for article in tqdm(articles):
    transformer_outputs = model.transformer(np.array([article["token_ids"]]))[0][0]
    cls_token = transformer_outputs[0].numpy().tolist()

    db["articles"].find_one_and_update(
        {"_id": article["_id"]},
        {"$set": {"cls_token": cls_token} }
    )