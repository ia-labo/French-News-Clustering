import sys
from copy import copy
from datetime import datetime, timedelta
from math import ceil
import threading

import utils
import numpy as np
from scipy.spatial.distance import cosine, seuclidean
from sklearn.metrics import jaccard_score

from model.micro_cluster import MicroCluster


class Denstream():
    def __init__(self, db, lambd=1, eps=1, cluster_weight=350, simulation=True, dt=None):
        """
            DenStream - Density-Based Clustering over an Evolving Data Stream with
            Noise.

            Parameters
            ----------
            lambd: float, optional
                The forgetting factor. The higher the value of lambda, the lower
                importance of the historical data compared to more recent data.
            eps : float, optional
                The maximum distance between two samples for them to be considered
                as in the same neighborhood.
            cluster_weight: integer, optional
                Minimum weight for cluster to be considered as potential
            simulation: boolean, optional
                If simulation is True timestamp are retrieved from articles and not
                in real time.
            dt: datetime
                Start datetime if simulation mode

            References
            ----------
            Feng Cao, Martin Estert, Weining Qian, and Aoying Zhou. Density-Based
            Clustering over an Evolving Data Stream with Noise.
        """
        self.db = db
        self.lambd = lambd
        self.eps = eps
        self.p_micro_clusters = []
        self.o_micro_clusters = []
        self.simulation = simulation
        self.cluster_weight = cluster_weight
        self.mandatory_entities_treshold = 10
        self.min_split_entities_in_clusters = 5

        if self.simulation:
            self.t = dt.timestamp()
        else:
            self.t = datetime.now().timestamp()
        self.decay()

    def reset_from_db(self):
        """
            Restart algorithm with clusters and articles in database
        """
        results = self.db["clusters"].aggregate([
            { "$match": { "active": True } },
            {
                "$lookup": {
                    "from": "articles",
                    "localField": "articles",
                    "foreignField": "_id",
                    "as": "article_list"
                }
            }
        ])

        for result in results:
            mc = MicroCluster.from_database(result, self.db)
            if result["type"] == "potential":
                self.p_micro_clusters.append(mc)
            else:
                self.o_micro_clusters.append(mc)

    def fit_article(self, article):
        """
            Method to add article to clustering algorithm
        """
        self._partial_fit(article)

    def _distance_function(self, micro_cluster, sample):
        """
            Compute distance function between a cluster and a sample
        """
        cosine_dist = cosine(micro_cluster.center(), sample.cls_token)
        jcd_similarity = utils.weighted_jaccard_similarity(sample, micro_cluster)
        category_dist = np.linalg.norm(micro_cluster.get_categories() - sample.categories)
        jcd_distance = 1 - jcd_similarity

        category_dist = category_dist ** 2

        stats = {
            "cosine_distance": cosine_dist,
            "jcd_distance": jcd_distance,
            "category_distance": category_dist,
        }

        if len(micro_cluster.entities) > 1 and jcd_distance == 1:
            return 1, stats
        elif jcd_distance == 0:
            return cosine_dist, stats
        return cosine_dist * jcd_distance * 2, stats

    def _get_cluster_entities(self, micro_cluster, article_entities):
        """
            Get entity name from a cluster as a set
            params:
                micro_clusters : List of clusters
                article_entities: set of article entities

            return:
                Set of entities name
                None if cluster has a mandatory entity which is not matched in this 
        """
        y_true_set = set()
        for entity, value in micro_cluster.entities.items():
            y_true_set.add(entity)
            # Check if entity is mandatory
            if value > self.mandatory_entities_treshold and entity not in article_entities:
                return None
        return y_true_set

    def _get_mosts_matching_clusters(self, sample, micro_clusters):
        """
            Get clusters with as much entities matching as possible
            params:
                sample : Current article
                micro_clusters : List of clusters

            return:
                List of boolean if a cluster should match
        """
        if len(micro_clusters) == 0 or len(sample.entities) <= 1:
            return [True] * len(micro_clusters)
        y_pred_set = set([entity["word"] for entity in sample.entities])

        mcs = []
        max_len = 0
        for micro_cluster in micro_clusters:
            y_true_set = self._get_cluster_entities(micro_cluster, y_pred_set)
            if y_true_set == None:
                mcs.append(-1)
            else:
                intersec_len = len(y_true_set.intersection(y_pred_set))
                mcs.append(intersec_len)
                if intersec_len > max_len:
                    max_len = intersec_len

        if max_len == 1:
            return [True] * len(micro_clusters)
        return [a == max_len for a in mcs]

    def _get_nearest_micro_cluster(self, sample, micro_clusters, valid_clusters):
        """
            Get nearest micro cluster form list of clusters.

            Return nearest micro cluster with its index
        """
        smallest_distance = sys.float_info.max
        nearest_micro_cluster = None
        nearest_micro_cluster_index = -1
        for idx, (micro_cluster, valid) in enumerate(zip(micro_clusters, valid_clusters)):
            if not valid:
                continue
            current_distance, _ = self._distance_function(micro_cluster, sample)
            if current_distance < smallest_distance:
                smallest_distance = current_distance
                nearest_micro_cluster = micro_cluster
                nearest_micro_cluster_index = idx

        return nearest_micro_cluster_index, nearest_micro_cluster

    def _try_merge(self, sample, micro_cluster, force=False):
        """
            Try to merge cluster and sample
            If distance is lower than epsilon merge, else do not touch.
        """
        if micro_cluster is not None:
            dist, stats = self._distance_function(micro_cluster, sample)
            if dist <= self.eps or force:
                self.db["event"].insert_one({
                    "type": "insert_sample",
                    "cluster": micro_cluster._id,
                    "sample": sample._id,
                    "distance": dist,
                    "stats": stats
                })
                micro_cluster.insert_sample(sample)
                return True
        return False

    def _new_potential_cluster(self, articles, cluster_list):
        """
            Insert pre-computed cluster with a list of articles in potential list.
        """
        if len(articles) == 0:
            return cluster_list

        micro_cluster = MicroCluster(self.lambd, self.db)
        micro_cluster.update_type("potential")
        for article in articles:
            micro_cluster.insert_sample(article)

        cluster_list.append(micro_cluster)
        return cluster_list

    def _split_disparate_cluster(self, cluster_list, cluster_index, disparate_entities):
        """
            Split clusters which contains disparate entities.

            Disparate entities are couple of entities that are in the same cluster
            but never in the same articles
        """
        entity_a, entity_b = disparate_entities

        articles_a = []
        articles_b = []
        articles_c = []
        for article in cluster_list[cluster_index].articles:
            if entity_a in [entity["word"] for entity in article.entities]:
                articles_a.append(article)
            elif entity_b in [entity["word"] for entity in article.entities]:
                articles_b.append(article)
            else:
                articles_c.append(article)

        new_cluster_list = []
        new_cluster_list = self._new_potential_cluster(articles_a, new_cluster_list)
        new_cluster_list = self._new_potential_cluster(articles_b, new_cluster_list)
        new_cluster_list = self._new_potential_cluster(articles_c, new_cluster_list)

        self.db["event"].insert_one({
            "type": "split_cluster_disparate",
            "old_cluster": cluster_list[cluster_index]._id,
            "new_clusters": [nc._id for nc in new_cluster_list]
        })

        cluster_list.extend(new_cluster_list)
        self.db["clusters"].delete_one({"_id": cluster_list[cluster_index]._id})
        del cluster_list[cluster_index]
        return new_cluster_list

    def _split_non_disparate_cluster(self, cluster_list, cluster_index):
        """
            Split a cluster without disparate entities.
            Explode a cluster and try to merge articles in algorithm again.
        """
        new_clusters = set([])
        articles = cluster_list[cluster_index].articles
        self.db["clusters"].delete_one({"_id": cluster_list[cluster_index]._id})
        del self.p_micro_clusters[cluster_index]

        for article in articles:
            new_clusters.add(self._merging(article))
        return list(new_clusters)

    def _merging(self, article):
        # Try to merge article with its nearest p_micro_cluster
        p_mcs = self._get_mosts_matching_clusters(article, self.p_micro_clusters)
        p_index, nearest_p_micro_cluster = self._get_nearest_micro_cluster(article, self.p_micro_clusters, p_mcs)
        success = self._try_merge(article, nearest_p_micro_cluster)

        if success:
            # Find if cluster contains disparate entities
            disparate_entities = nearest_p_micro_cluster.check_disparate_entities(self.min_split_entities_in_clusters)
            if disparate_entities:
                self._split_disparate_cluster(self.p_micro_clusters, p_index, disparate_entities)
            return nearest_p_micro_cluster
        else:
            # Try to merge article into its nearest o_micro_cluster
            o_mcs = self._get_mosts_matching_clusters(article, self.o_micro_clusters)
            o_index, nearest_o_micro_cluster = self._get_nearest_micro_cluster(article, self.o_micro_clusters, o_mcs)
            success = self._try_merge(article, nearest_o_micro_cluster)
            if success:
                if self._decay_function(nearest_o_micro_cluster) > self.cluster_weight:
                    del self.o_micro_clusters[o_index]
                    self.p_micro_clusters.append(nearest_o_micro_cluster)
                    nearest_o_micro_cluster.update_type("potential")
                    self.db["event"].insert_one({
                        "type": "outlier_to_potential",
                        "cluster": nearest_o_micro_cluster._id,
                    })
                    return nearest_o_micro_cluster
            else:
                # Create new o_micro_cluster
                micro_cluster = MicroCluster(self.lambd, self.db)
                micro_cluster.insert_sample(article)
                self.o_micro_clusters.append(micro_cluster)
                self.db["event"].insert_one({
                    "type": "create_outlier",
                    "cluster": micro_cluster._id,
                })
                return micro_cluster
        return None

    def _partial_fit(self, sample):
        self._merging(sample)
        if self.simulation:
            self.t = sample.created_at
        else:
            self.t = datetime.now().timestamp()

    def _decay_function(self, micro_cluster):
        """
            Compute weight loss over time
        """
        return ((micro_cluster.get_weight() * 60 + micro_cluster.created_at) - self.t) / 60

    def decay(self):
        for idx, micro_cluster in enumerate(self.p_micro_clusters):
            if not self._decay_function(micro_cluster) >= 1:
                self.p_micro_clusters[idx].deactivate()
                del self.p_micro_clusters[idx]
        for idx, micro_cluster in enumerate(self.o_micro_clusters):
            if not self._decay_function(micro_cluster) >= 1:
                self.o_micro_clusters[idx].deactivate()
                del self.o_micro_clusters[idx]
        threading.Timer(60, self.decay).start()

    def move_article(self, article, cluster_from, cluster_to):
        """
            Move article from cluster

            Parameters :
                cluster_from: Origin cluster
                cluster_to: Destination cluster
        """
        from_cluster = None
        to_cluster = None

        # Check potential clusters
        for idx, cluster in enumerate(self.p_micro_clusters):
            if cluster._id == cluster_from["_id"]:
                from_cluster = idx
            if cluster._id == cluster_to["_id"]:
                to_cluster = idx

        if from_cluster == None or to_cluster == None:
            raise ValueError

        # Remove article from old cluster
        if len(self.p_micro_clusters[from_cluster].articles) > 1:
            self.p_micro_clusters[from_cluster].remove_sample(article)
        else:
            self.p_micro_clusters[from_cluster].deactivate()
            del self.p_micro_clusters[from_cluster]

        # Insert into new cluster
        self._try_merge(article, self.p_micro_clusters[to_cluster], force=True)

    def split_cluster(self, cluster_id):
        """
            Find cluster and split it either with its disparate entities or by exploding it.
        """
        for idx, cluster in enumerate(self.p_micro_clusters):
            if str(cluster._id) == cluster_id:
                disparate_entities = cluster.check_disparate_entities(self.min_split_entities_in_clusters)
                if disparate_entities:
                    clusters = self._split_disparate_cluster(self.p_micro_clusters, idx, disparate_entities)
                else:
                    clusters = self._split_non_disparate_cluster(self.p_micro_clusters, idx)
                return True, clusters

        for idx, cluster in enumerate(self.o_micro_clusters):
            if cluster._id == cluster_id:
                disparate_entities = cluster.check_disparate_entities(self.min_split_entities_in_clusters)
                if disparate_entities:
                    clusters = self._split_disparate_cluster(self.p_micro_clusters, idx, disparate_entities)
                else:
                    clusters = self._split_non_disparate_cluster(self.p_micro_clusters, idx)
                return True, clusters
        return False, []
