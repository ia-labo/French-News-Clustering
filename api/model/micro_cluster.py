import json
import bson
import utils
from math import trunc
import numpy as np
import pymongo
from datetime import datetime

from .article import Article


class MicroCluster():
    """
        MicroCluster - Cluster for DenStream algorithm

        Parameters
        ----------
        lambd: float
            The forgetting factor.
        db : pymongo collection
            The maximum distance between two samples for them to be considered
            as in the same neighborhood.
    """
    def __init__(self, lambd, db, _id=None):
        self.db = db
        self.lambd = lambd
        self.weight = 0
        self.created_at = None
        self.updated_at = None
        self.articles = []
        self.entities = {}
        self.categories = []
        self._type = "outlier"
        self.active = True

        if not _id:
            res = self.db["clusters"].insert_one({
                "active": True,
                "type": self._type,
                "lambd": self.lambd,
            })
            self._id = res.inserted_id
        else:
            self._id = _id

    def update_type(self, _type):
        if _type not in ["potential", "outlier"]:
            raise ValueError

        self._type = _type
        self.db["clusters"].find_one_and_update(
            {"_id": self._id},
            {"$set": {"type": self._type} }
        )

    def deactivate(self):
        self.db["clusters"].find_one_and_update(
            {"_id": self._id},
            {
                "$set": {
                    "active": False,
                    "closed_at": self.created_at + self.weight * 60,
                }
            }
        )
        self.db["event"].insert_one({
            "type": "deactivate_" + self._type,
            "cluster": self._id,
        })
        self.active = False

    def _insert_entities(self, entities, minimum_occurence):
        entities = list(set([entity["word"] for entity in entities]))

        for entity in entities:
            if entity in self.entities.keys():
                self.entities[entity] += 1
            elif minimum_occurence == 0:
                self.entities[entity] = 1
            else:
                # If entity must already exists, check all articles to find if any contain the same entity
                count = 0
                for article in self.articles:
                    for article_entity in article.entities:
                        if article_entity["word"] == entity:
                            count += 1
                            continue # Do not count double occurence in same article
                if count >= minimum_occurence:
                    self.entities[entity] = count

    def check_disparate_entities(self, minimum_disparate):
        # Get tuple list of entities which occur more than `minimum_disparate` times
        important_entities = [entity for entity, value in self.entities.items() if value > minimum_disparate]
        important_entities = utils.pair_list(important_entities)

        entity_pair_in_list = False
        for entity_1, entity_2 in important_entities:
            # Check if both entities appears in the same article
            for article in self.articles:
                article_entity_list = [entity["word"] for entity in article.entities]
                if entity_1 in article_entity_list and entity_2 in article_entity_list:
                    entity_pair_in_list = True
                    break
            
            # If entity are never seen together, they are disparate
            if not entity_pair_in_list:
                return (entity_1, entity_2)
        return None

    def _find_and_update(self):
        self.db["clusters"].find_one_and_update(
            {"_id": self._id},
            {
                "$set": {
                    "weight": self.weight,
                    "articles": [article._id for article in self.articles],
                    "entities": self.entities,
                    "created_at": self.created_at,
                    "updated_at": self.updated_at,
                },
            }
        )

    def insert_sample(self, article):
        if self.weight != 0:
            # Avoid doublons
            for news in self.articles:
                if news == article:
                    return

            self.weight += 60 / (1 + self.lambd * (len(self.articles) - 1))

            self.articles.append(article)
            self.updated_at = datetime.now().timestamp()
            self.categories.append(article.categories)
            self._insert_entities(article.entities, trunc(len(self.entities) / 5))
        else:
            self.weight = 5 * 60 # 5 hours
            self.articles = [article]
            if len(article.entities) > 0:
                self.entities = { entity["word"]: 1 for entity in article.entities}
            self.created_at = article.created_at
            self.categories = [article.categories]
        self._find_and_update()

    def remove_sample(self, article):
        self.weight -= 60 / (1 + self.lambd * (len(self.articles) - 1))

        article_id = -1
        for idx, art in enumerate(self.articles):
            if article._id == art._id:
                article_id = idx
                break
        if article_id == -1:
            return

        del self.articles[article_id]
        del self.categories[article_id]

        self.updated_at = datetime.now().timestamp()
        if len(article.entities) > 0:
            for entity in article.entities:
                entity = entity["word"]
                if entity not in self.entities.keys():
                    continue
                if self.entities[entity] > 1:
                    self.entities[entity] -= 1
                else:
                    self.entities.pop(entity)
        self.db["clusters"].find_one_and_update(
            {"_id": self._id},
            {
                "$set": {
                    "weight": self.weight,
                    "articles": [article._id for article in self.articles],
                    "entities": self.entities,
                    "created_at": self.created_at,
                    "updated_at": self.updated_at,
                },
            }
        )

    def center(self):
        return np.mean([article.cls_token for article in self.articles], axis=0)

    def get_categories(self):
        return np.mean(self.categories, axis=0)

    def get_weight(self):
        return self.weight

    def __str__(self):
        return "Micro cluster " + str(self._id) + " : " + "; ".join([article.raw for article in self.articles])

    @classmethod
    def from_database(cls, cluster, db):
        mc = cls(cluster["lambd"], db, _id=cluster["_id"])
        mc.weight = cluster["weight"]
        mc.created_at = cluster["created_at"]
        mc.updated_at = cluster["updated_at"]
        mc.articles = [Article.from_database(db, article) for article in cluster["article_list"]]
        mc.entities = cluster["entities"]
        mc.categories = [article.categories for article in mc.articles]
        mc._type = cluster["type"]
        mc.active = cluster["active"]
        return mc

    def toJson(self, category_labels):
        def timestamp_to_string(dt):
            return datetime.fromtimestamp(dt).strftime("%Y-%m-%dT%H:%M")

        return {
            "_id": str(self._id),
            "lambd": self.lambd,
            "active": self.active,
            "categories": [category_labels[idx] for idx, c in enumerate(self.get_categories()) if c > 0.9],
            "entities": self.entities,
            "created_at": timestamp_to_string(self.created_at),
            "updated_at": timestamp_to_string(self.updated_at) if isinstance(int, datetime) else timestamp_to_string(self.created_at),
            "weight": self.weight,
            "articles": [article.toJson(category_labels) for article in self.articles]
        }
