import json
import threading
from datetime import datetime, timedelta

import bson
import flask_monitoringdashboard as dashboard
import numpy as np
import pymongo
import tensorflow as tf
from flask import Flask, jsonify, request
from transformers import FlaubertTokenizer
from transformers.modeling_tf_flaubert import TFFlaubertForSequenceClassification

import utils
from denstream import Denstream
from flaubert_token_classification import TFFlaubertForTokenClassification
from model.article import Article
from model.micro_cluster import MicroCluster


app = Flask(__name__, static_url_path='',  template_folder='static')
dashboard.config.init_from(file='config/dashboard.cfg')
dashboard.bind(app)

models = {
    "ner": TFFlaubertForTokenClassification.from_pretrained("../models/ner"),
    "pos": TFFlaubertForTokenClassification.from_pretrained("../models/pos"),
    "categorisation": TFFlaubertForSequenceClassification.from_pretrained("../models/categorisation")
}
tokenizer = FlaubertTokenizer.from_pretrained("jplu/tf-flaubert-base-cased")
SEQUENCE_LENGTH = 64

categories = ['culture', 'france', 'international', 'santé', 'science_high-tech', 'sports', 'économie']
ner_labels = ["LOC", "MISC", "ORG", "PER", "O"]
pos_labels = ["_", "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"]

client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["bert_clustering"]

dt = datetime(year=2020, month=4, day=27, hour=0, minute=0, second=1)
clusters = Denstream(db, eps=0.01, lambd=0.1, simulation=True, dt=dt)
clusters.reset_from_db()


@app.errorhandler(404)
def page_not_found(e):
    return json.loads("{}"), 404


# Processing routes
@app.route('/preprocessing/category', methods=["POST"])
def categorisation():
    '''
        Get the category of a news title.
    '''
    if not "title" in request.json:
        return json.loads("{}"), 400

    title = utils.preprocess_title(request.json["title"])
    input_tokens = tokenizer.encode_plus(
        title,
        max_length=SEQUENCE_LENGTH,
        pad_to_max_length=SEQUENCE_LENGTH,
        add_special_tokens=True,
        return_tensors='tf',
    )

    out = models["categorisation"](input_tokens["input_ids"])
    idx = int(np.argmax(out[0]))
    return jsonify({
        "title": title,
        "category": categories[idx],
        "results": { key:val for key, val in zip(categories, out[0].numpy().tolist()[0]) },
    }), 200


@app.route('/preprocessing/pos_tag', methods=["POST"])
def pos_tag():
    '''
        Detect part-of-speech tagging
    '''
    if not "title" in request.json:
        return json.loads("{}"), 400

    title = utils.preprocess_title(request.json["title"])
    sentence = tokenizer.tokenize(title)
    input_tokens = tokenizer.encode_plus(
        title,
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

    preductions = models["pos"](input_tokens["input_ids"], **inputs)
    preductions = np.argmax(preductions, axis=2)[0][0]
    entities = utils.post_token_classification(sentence, preductions, pos_labels)

    return jsonify({
        "title": title,
        "results": entities,
    }), 200


@app.route('/preprocessing/ner', methods=["POST"])
def ner():
    '''
        Detect entities depending on a category.
    '''
    if not "title" in request.json:
        return json.loads("{}"), 400

    title = utils.preprocess_title(request.json["title"])
    sentence = tokenizer.tokenize(title)
    input_tokens = tokenizer.encode_plus(
        title,
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
    ner_output = models["ner"](input_tokens["input_ids"], **inputs)[0]
    ner_output = np.argmax(ner_output, axis=2)[0]
    entities = utils.post_token_classification(sentence, ner_output, ner_labels, normalize=True)

    return jsonify({
        "title": title,
        "results": entities,
    }), 200


@app.route('/articles', methods=["GET"])
def get_article_list():
    '''
        Get all articles since date
    '''
    from_date, err = utils.date_from_request(request, "from")
    if err:
        return err, 400
    to_date, err = utils.date_from_request(request, "to", allow_missing=True)
    if err:
        return err, 400

    results = db["articles"].find({ "created_at": { "$gte": from_date.timestamp(), "$lt": to_date.timestamp() } } )

    articles = []
    for result in results:
        articles.append(Article.from_database(db, result).toJson(categories))

    return jsonify({
        "articles": articles
    })


@app.route('/article', methods=["GET"])
def get_article():
    article_id = request.args.get("_id")
    if article_id == None:
        return jsonify({"error": "You must set an article id"}), 400

    result = db["articles"].find_one({"_id": bson.ObjectId(oid=article_id)})
    if result:
        return jsonify(Article.from_database(db, result).toJson(categories))
    else:
        return jsonify({"error": "No article found with this id"}), 400


@app.route('/article', methods=["DELETE"])
def delete_article():
    article_id = request.args.get("_id")
    if article_id == None:
        return jsonify({"error": "You must set an article id"}), 400

    article_id = bson.ObjectId(oid=article_id)

    article = db["articles"].find_one({"_id": article_id})
    if article:
        db["articles"].delete_one({"_id": article_id})

        cluster = db.clusters.find_one({ "articles": { "$elemMatch": { "$eq": article_id } }})
        if len(cluster["article"]) == 1:
            # Remove cluster if it contains only 1 article
            db["clusters"].delete_one({"_id": cluster["_id"]})
        else:
            # Update cluster otherwise
            db.clusters.update(
                filter={ "articles": { "$elemMatch": {"$eq": article_id}}},
                update={ "$pull": { "articles": article_id}},
                multi= True
            )
        return jsonify({"success": "Article removed from database"}), 200
    else:
        return jsonify({"error": "No article found with this id"}), 400


@app.route('/article/move', methods=["POST"])
def move_article():
    article_id =  request.json["article_id"]
    cluster_from =  request.json["cluster_from"]
    cluster_to =  request.json["cluster_to"]

    if not article_id or not cluster_from or not cluster_to:
        return jsonify({"error": "argument `article_id`, `cluster_from` and `cluster_to` must be in JSON input"}), 400

    # Get info from database
    result = db["articles"].find_one({ "_id": bson.ObjectId(oid=article_id)})
    cluster_from = db["clusters"].find_one({ "_id": bson.ObjectId(oid=cluster_from)})
    cluster_to = db["clusters"].find_one({ "_id": bson.ObjectId(oid=cluster_to)})
    if not cluster_from or not cluster_to or not result:
        return jsonify({"error": "An id is invalid"}), 400

    article = Article.from_database(db, result)
    transformer_outputs = models["categorisation"].transformer(np.array([article.token_ids]).reshape(1, -1))[0][0]
    article.cls_token = transformer_outputs[0].numpy().tolist()

    clusters.move_article(article, cluster_from, cluster_to)
    return jsonify({"success": True}), 200


@app.route('/clustering', methods=["POST"])
def cluster_article():
    '''
        Cluster news article
    '''
    if not "title" in request.json:
        return json.loads("{}"), 400

    title = utils.preprocess_title(request.json["title"])
    sentence = tokenizer.tokenize(title)
    input_tokens = tokenizer.encode_plus(
        title,
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
    ner_output = models["ner"](input_tokens["input_ids"], **inputs)[0]
    ner_output = np.argmax(ner_output, axis=2)[0]
    entities = utils.post_token_classification(sentence, ner_output, ner_labels, normalize=True)

    # Get categorisation
    cat_predictions = models["categorisation"](input_tokens["input_ids"])

    # Get CLS token
    transformer_outputs = models["categorisation"].transformer(input_tokens["input_ids"])[0][0]

    article = Article(
        db,
        raw=title,
        token_ids=input_tokens["input_ids"].numpy().tolist(),
        cls_token=transformer_outputs[0].numpy().tolist(),
        created_at=request.json["timestamp"],
        categories=utils.sigmoid(cat_predictions[0][0].numpy()).tolist(),
        entities=entities
    )
    clusters.fit_article(article)
    return jsonify({"success": True})


@app.route('/clustering', methods=["GET"])
def clustering_list():
    '''
        Get active clusters
    '''
    clusters_from, err = utils.date_from_request(request, "from")
    if err:
        return err, 400
    clusters_to, err = utils.date_from_request(request, "to")
    if err:
        return err, 400

    cursor = db["clusters"].aggregate([
        {
            "$match": {
                "type": "potential",
                "$or": [
                    {
                        # Closed between begin and end
                        "closed_at":
                        {
                            "$gte": clusters_from.timestamp(),
                            "$lte": clusters_to.timestamp(),
                        }
                    },
                    {
                        # Created between begin and end
                        "created_at":
                        {
                            "$gte": clusters_from.timestamp(),
                            "$lte": clusters_to.timestamp(),
                        }
                    },
                    {
                        # Created after begin date and still active
                        "created_at": { "$gte": clusters_from.timestamp() },
                        "active": True
                    },
                    {
                        # Created before and close after, so active between the time delta
                        "created_at": { "$lte": clusters_from.timestamp() },
                        "closed_at": { "$gte": clusters_to.timestamp() },
                    },
                ],
            }
        },
        {
            "$lookup": {
                "from": "articles",
                "localField": "articles",
                "foreignField": "_id",
                "as": "article_list"
            }
        }
    ])

    clstrs = []
    for cluster in cursor:
        clstrs.append(MicroCluster.from_database(cluster, db).toJson(categories))

    return jsonify(clstrs)


@app.route('/split_clustering', methods=["POST"])
def split_clustering():
    '''
        Split active cluster by ID
    '''
    cluster_id = request.json["cluster_id"]
    valid, new_clusters = clusters.split_cluster(cluster_id)
    return jsonify({"success": valid, "new_clusters": [cl.toJson(categories) for cl in new_clusters]})


def batch_fit(articles):
    for article in articles:
        clusters.fit_article(article)


@app.route('/cluster_from_database', methods=["POST"])
def cluster_from_database():
    '''
        Recreate clusters
    '''
    # Remove clusters
    db["clusters"].drop()
    db["event"].drop()
    clusters.p_micro_clusters = []
    clusters.o_micro_clusters = []
    if clusters.simulation:
        clusters.t = dt.timestamp()

    # Get all articles
    results = db["articles"].find()

    articles = []
    for result in results:
        articles.append(Article.from_database(db, result))
    threading.Thread(target=batch_fit, args=([articles])).start()

    return jsonify({"success": True})


if __name__ == '__main__':
    app.run()
