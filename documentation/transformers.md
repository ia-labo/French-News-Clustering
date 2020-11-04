# Transformers

## Librairie

[Hugging face](https://huggingface.co/) [Transformers](https://github.com/huggingface/transformers) est une librairie basée sur Tensorflow et Pythorch qui propose une implémentation de [l'architecture Transformer](https://arxiv.org/pdf/1706.03762.pdf).

Ils donnent accès a des modèles à l'état de l'art du traitement automatisé du langage naturel (TALN ou NLP pour Natural Language Processing en anglais) comme BERT, XLNet ou Camembert et Flaubert que nous utiliserons dans ce projet.

## Modèles

### [Camembert](https://camembert-model.fr/)

CamemBERT est un modèle de langage ([Language Model](https://en.wikipedia.org/wiki/Language_model)) à l'état de l'art sur plusieurs tâches de NLP. Il est basé sur [l'architecture RoBERTa](https://ai.facebook.com/blog/roberta-an-optimized-method-for-pretraining-self-supervised-nlp-systems/) et entrainé sur la version française du corpus OSCAR.


#### Pourquoi nous n'utilisons pas Camembert ?

L'algorithme Camembert a des problèmes d'apprentissage avec la librairie Tensorflow et 

- [Issue Github](https://github.com/huggingface/transformers/issues/3361)
- [Post stack overflow](https://stackoverflow.com/questions/60761761/hugging-face-transformer-classifier-fail-on-imbalance-dataset) : Caché car supprimé par SO mais similaire à l'issue Github


### [Flaubert](https://arxiv.org/pdf/1912.05372.pdf)

FlauBERT est une version française de BERT entrainé sur un corpus très large et hétérogène.
