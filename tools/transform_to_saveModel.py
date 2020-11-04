import tensorflow as tf

from transformers import (FlaubertConfig, FlaubertTokenizer)
from flaubert_token_classification import TFFlaubertForTokenClassification


strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")

with strategy.scope():
    model = TFFlaubertForTokenClassification.from_pretrained("./models/ner")#, config=config)
    #tokenizer = FlaubertTokenizer.from_pretrained("jplu/tf-flaubert-base-cased")
    
tf.saved_model.save(model, "models/savedModel/ner/1/")
