from transformers.modeling_tf_xlm import TFXLMPreTrainedModel
from transformers.modeling_tf_flaubert import TF_FLAUBERT_PRETRAINED_MODEL_ARCHIVE_MAP, TFXLMMainLayer, TFFlaubertMainLayer
from transformers.configuration_flaubert import FlaubertConfig
from transformers.modeling_tf_utils import TFPreTrainedModel, get_initializer
import tensorflow as tf

class TFXLMForTokenClassification(TFXLMPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels

        self.transformer = TFXLMMainLayer(config, name="transformer")
        self.dropout = tf.keras.layers.Dropout(config.dropout)
        self.classifier = tf.keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.init_std), name="classifier"
        )

    def call(self, inputs, **kwargs):
        transformer_outputs = self.transformer(inputs, **kwargs)
        sequence_output = transformer_outputs[0]

        sequence_output = self.dropout(sequence_output, training=kwargs.get("training", False))
        logits = self.classifier(sequence_output)

        outputs = (logits,) + transformer_outputs[1:]  # Keep new_mems and attention/hidden states if they are here
        return outputs

class TFFlaubertForTokenClassification(TFXLMForTokenClassification):
    config_class = FlaubertConfig
    pretrained_model_archive_map = TF_FLAUBERT_PRETRAINED_MODEL_ARCHIVE_MAP

    def __init__(self, config, *inputs, **kwargs):
        super(TFFlaubertForTokenClassification, self).__init__(config, *inputs, **kwargs)
        self.transformer = TFFlaubertMainLayer(config, name="transformer")