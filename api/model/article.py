import json
from datetime import datetime

import bson
import numpy as np
import pymongo


class Article:
    def __init__(self, db, raw, cls_token, token_ids, created_at, categories, entities=None, _id=None):
        self.raw = raw
        self.cls_token = cls_token
        self.token_ids = token_ids[0]
        self.entities = entities
        self.categories = categories
        self._set_creation_date(created_at)

        if _id:
            self._id = _id
        else:
            res = db["articles"].insert_one({
                "raw": self.raw,
                "token_ids": self.token_ids,
                "created_at": self.created_at,
                "entities": self.entities,
                "categories": self.categories,
                "cls_token": self.cls_token,
            })
            self._id = res.inserted_id

    def _set_creation_date(self, created_at):
        if isinstance(created_at, float) or isinstance(created_at, int):
            self.created_at = created_at
        elif isinstance(created_at, str):
            self.created_at = datetime.strptime(created_at, "%Y-%m-%d %H:%M:%S").timestamp()
        else:
            raise ValueError

    def get_entities(self):
        return self.entities

    @classmethod
    def from_database(cls, db, article):
        return cls(
            db,
            article["raw"],
            article["cls_token"],
            article["token_ids"],
            article["created_at"],
            article["categories"],
            article["entities"],
            article["_id"]
        )

    def toJson(self, category_labels):
        return {
            "_id": str(self._id),
            "raw": self.raw,
            "entities": self.entities,
            "categories": [category_labels[idx] for idx, c in enumerate(self.categories) if c > 0.9],
            "created_at": datetime.fromtimestamp(self.created_at).strftime("%Y-%m-%dT%H:%M"),
        }

