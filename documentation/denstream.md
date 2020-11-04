# Denstream

[Denstream papier](https://archive.siam.org/meetings/sdm06/proceedings/030caof.pdf)

## Algorithm

L'algorithme de clustering utilisé dans ce projet est basé sur Denstream. Il utilise le système de double list `outliers` et `potentials`. Cet algorithme est particulièrement adapté aux flux de données.

- Les clusters `outliers` sont composés d'articles trop peu représentés pour être déclaré comme un sujet.
- Les clusters `potentials` représente des sujets d'actualités.

## Modifier l'algorithme

Pour modifier rapidement l'algorithme, on peut jouer avec les paramêtres disponible dans la classe `Denstream`:
- Epsilon (eps) : Distance maximale pour considérer un article comme membre d'un cluster.
- Lambda (lambd) : Facteur d'oublie, permet d'accorder moins d'importance aux articles qui rentre dans un cluster.
- cluster_weight : Poids minimal d'un cluster `outlier` pour être considéré `potential`
- simulation : Booléen, si vrai l'horloge interne de l'algorithme se base sur la date du dernier article ajouté.
- dt : Datetime, si simulation est vrai, dt doit représenter une heure.
- mandatory_entities_treshold : Nombre d'entités dans un cluster à partir duquel celle-ci devient obligatoire pour entrer dans le groupe. Ce chiffre est volontairement important. Exemple :
    - Si mandatory_entities_treshold = 10
    - A partir de 10 entités X dans le cluster C
    - Chaque nouveau article A, doit posséder une entité X pour rentrer dans C.
- min_split_entities_in_clusters : Quand un cluster devient trop gros, il devient souvent "flou" et englobe beaucoup d'articles dans un périmètre sémantique large. Pour remédier à ça, on 

## Optimisations

Liste des optimisations et modifications appliqués à l'algorithme pour qu'il soit plus adapté à notre problème.

- Dans le papier une deuxième couche de clustering (DBSCAN) est appliqué sur les clusters outliers. Cette couche est adapté pour créer des meta-clusters et ainsi comprendre des informations plus généralistes qui peuvent être intéressantes sur des masses de données très importantes. Dans notre cas, nous ne cherchons pas obtenir ce type d'information mais seulement à regrouper les articles par sujet, ça ne parrait donc pas nécessaire.
- La mesure de distance utilisé originellement dans Denstream est la distance euclidienne. Dans notre cas nous utiliserons la distance cosine qui est beaucoup plus adapté pour comprendre les mots qui ont une signification proche. A cela nous multiplions, si possible, une distance de jaccard représentative du nombre d'entités en commun entre l'article et le cluster.
- La distance de jaccard est pondéré par le nombre d'occurence dans le cluster.
- Methode de distance temporelle modifié :
    - Quand un cluster est créer, il dispose de 5 heures de vies.
    - Ensuite, pour chaque article ajouté, on incrémente de 60 minutes moins un facteur d'oublie (`lambda`)
