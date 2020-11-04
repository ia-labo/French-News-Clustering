# Datasets

## 1. wikiNER

### **Description**

[WikiNer](https://figshare.com/articles/Learning_multilingual_named_entity_recognition_from_Wikipedia/5462500) est un dataset multi-lingue regroupant des entités extraites depuis Wikipedia.
Les modèles français de [Spacy ont était entrainés](https://spacy.io/models/fr) avec ce dataset.

### **Entités**

Il est composé de 4 entités :
- LOC : Géolocalisation
- ORG : Organisation (état, entreprise, )
- PER : Personne
- MISC : Autres


## 2. GSD

[GSD](https://universaldependencies.org/treebanks/fr_gsd/) est un dataset compris dans le framework [Universal depencies](https://universaldependencies.org/). Il est utilisé pour [l'étiquettage morpho synaxique](https://fr.wikipedia.org/wiki/%C3%89tiquetage_morpho-syntaxique) ou part-of-speech tagging.

### **Tags**

Il est composé de 21 tags principaux. Ces tags peuvent contenir des informations telle que la pluralité ou la singularité d'un mot mais aussi son genre. Seul les tags principaux ont était utilisé dans notrre algorithme.

- ADJ
- ADP
- ADV
- AUX
- CCONJ
- DET
- INTJ
- NOUN
- NUM
- PART
- PRON
- PROPN
- PUNCT
- SCONJ
- SYM
- VERB
- X
- _