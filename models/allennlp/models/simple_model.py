
import collections
import logging
from copy import deepcopy
from typing import Dict, List, Optional
import pdb 

from overrides import overrides
import torch

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules.transformer import (
    TextEmbeddings,
    ImageFeatureEmbeddings,
    BiModalEncoder,
    TransformerPooler,
)
from allennlp.nn import util

from transformers import AutoModel

logger = logging.getLogger(__name__)

@Model.register("simple_debug")
@Model.register("simple_debug_from_huggingface", constructor="from_huggingface_model_name")
class SimpleDebugModel(Model):
    """
    Simple MLP for debugging
    """
    def __init__(
        self,
        vocab: Vocabulary,
        text_embeddings: TextEmbeddings,
        image_embeddings: ImageFeatureEmbeddings,
        encoder: BiModalEncoder,
        pooled_output_dim: int,
        fusion_method: str = "sum",
        dropout: float = 0.1,
        label_namespace: str = "answers",
    ) -> None:
        super().__init__(vocab)

        self.loss = torch.nn.BCELoss()
        self.consistency_wrong_map: Dict[str, int] = collections.Counter()
        from allennlp.training.metrics import F1MultiLabelMeasure
        
        self.f1_metric = F1MultiLabelMeasure(average="micro")
        from allennlp.training.metrics.vqa import VqaMeasure

        self.vqa_metric = VqaMeasure()
        self.fusion_method = fusion_method

        self.embeddings = text_embeddings
        self.image_embeddings = image_embeddings
        self.encoder = encoder


        num_labels = vocab.get_vocab_size(label_namespace)
        self.label_namespace = label_namespace

        self.classifier = torch.nn.Linear(self.encoder.out_dim, num_labels)
        self.dropout = torch.nn.Dropout(dropout)
        self.loss = torch.nn.CrossEntropyLoss()

    @classmethod
    def from_huggingface_model_name(
        cls,
        vocab: Vocabulary,
        model_name: str,
        image_feature_dim: int,
        image_num_hidden_layers: int,
        image_hidden_size: int,
        image_num_attention_heads: int,
        image_intermediate_size: int,
        image_attention_dropout: float,
        image_hidden_dropout: float,
        image_biattention_id: List[int],
        image_fixed_layer: int,
        text_biattention_id: List[int],
        text_fixed_layer: int,
        combined_hidden_size: int,
        combined_num_attention_heads: int,
        pooled_output_dim: int,
        pooled_dropout: float = 0.1,
        fusion_method: str = "sum",
    ):
        transformer = AutoModel.from_pretrained(model_name)

        # TODO(mattg): This call to `transformer.embeddings` works with some transformers, but I'm
        # not sure it works for all of them, or what to do if it fails.
        # We should probably pull everything up until the instantiation of the image feature
        # embedding out into a central "transformers_util" module, or something, and just have a
        # method that pulls an initialized embedding layer out of a huggingface model.  One place
        # for this somewhat hacky code to live, instead of having to duplicate it in various models.
        text_embeddings = deepcopy(transformer.embeddings)

        # Albert (and maybe others?) has this "embedding_size", that's different from "hidden_size".
        # To get them to the same dimensionality, it uses a linear transform after the embedding
        # layer, which we need to pull out and copy here.
        if hasattr(transformer.config, "embedding_size"):
            config = transformer.config

            from transformers.modeling_albert import AlbertModel

            if isinstance(transformer, AlbertModel):
                linear_transform = deepcopy(transformer.encoder.embedding_hidden_mapping_in)
            else:
                logger.warning(
                    "Unknown model that uses separate embedding size; weights of the linear "
                    f"transform will not be initialized.  Model type is: {transformer.__class__}"
                )
                linear_transform = torch.nn.Linear(config.embedding_dim, config.hidden_dim)

            # We can't just use torch.nn.Sequential here, even though that's basically all this is,
            # because Sequential doesn't accept *inputs, only a single argument.

            class EmbeddingsShim(torch.nn.Module):
                def __init__(self, embeddings: torch.nn.Module, linear_transform: torch.nn.Module):
                    super().__init__()
                    self.linear_transform = linear_transform
                    self.embeddings = embeddings

                def forward(self, *inputs, **kwargs):
                    return self.linear_transform(self.embeddings(*inputs, **kwargs))

            text_embeddings = EmbeddingsShim(text_embeddings, linear_transform)

        image_embeddings = ImageFeatureEmbeddings(
            feature_dim=image_feature_dim,
            hidden_dim=image_hidden_size,
            dropout=image_hidden_dropout,
        )

        encoder = SimpleEncoder(
            pretrained_module=transformer,
            in_dim=1280,
            # in_dim=128,
            hidden_dim=32,
            num_layers=2, 
            out_dim=32
        )
        return cls(
            vocab=vocab,
            text_embeddings=text_embeddings,
            image_embeddings=image_embeddings,
            encoder=encoder,
            pooled_output_dim=pooled_output_dim,
            fusion_method=fusion_method,
            dropout=pooled_dropout,
        )

    @overrides
    def forward(
        self,  # type: ignore
        box_features: torch.Tensor,
        box_coordinates: torch.Tensor,
        question: TextFieldTensors,
        labels: Optional[torch.Tensor] = None,
        label_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size, _, feature_size = box_features.size()

        # TODO(mattg): have this make fewer assumptions.
        input_ids = question["tokens"]["token_ids"]
        token_type_ids = question["tokens"]["type_ids"]
        attention_mask = question["tokens"]["mask"]


        with torch.no_grad():
            # (batch_size, num_tokens, embedding_dim)
            embedding_output = self.embeddings(input_ids, token_type_ids)
            num_tokens = embedding_output.size(1)


            # (batch_size, num_boxes, image_embedding_dim)
            v_embedding_output = self.image_embeddings(box_features, box_coordinates)

        encoded = self.encoder(
            embedding_output,
            v_embedding_output)



        logits = self.classifier(encoded)
        probs = torch.softmax(logits, dim=-1)

        outputs = {"logits": logits, "probs": probs}
        if labels is not None and label_weights is not None:
            label_mask = labels > 1  # 0 is padding, 1 is OOV, which we want to ignore

            weighted_labels = util.masked_index_replace(
                logits.new_zeros(logits.size() + (1,)),
                labels.clamp(min=0),
                label_mask,
                label_weights.unsqueeze(-1),
            ).squeeze(-1)

            # weighted_labels now has shape (batch_size, num_labels).  We need to ignore the first
            # two columns of this in our loss function and accuracy metric.  The first column is a
            # padding label, and the second column is an OOV label.  We want the loss function to
            # be computed on every other label.
            binary_label_mask = weighted_labels.new_ones(logits.size())
            binary_label_mask[:, 0] = 0
            binary_label_mask[:, 1] = 0

            outputs["loss"] = (
                torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, weighted_labels, weight=binary_label_mask, reduction="sum"
                )
                / batch_size
            )
 
            self.f1_metric(logits, weighted_labels, binary_label_mask.bool())
            self.vqa_metric(logits, labels, label_weights)
        return outputs

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        result = self.f1_metric.get_metric(reset)
        result["vqa_score"] = self.vqa_metric.get_metric(reset)["score"]
        return result

class SimpleEncoder(torch.nn.Module):
    """
    really basic MLP class for overfitting, to test if dataloader is working 
    """
    def __init__(self, 
                 pretrained_module,
                 in_dim,
                 hidden_dim,
                 num_layers,
                 out_dim):
        super(SimpleEncoder, self).__init__() 
        self.pretrained_module = pretrained_module
        layers = [torch.nn.Linear(in_dim, hidden_dim)] + \
                    [torch.nn.Linear(hidden_dim, hidden_dim) for i in range(num_layers-2)] + \
                 [torch.nn.Linear(hidden_dim, out_dim)]
        network = []
        for i in range(len(layers)):
            network.append(layers[i])
            if i < len(layers)-1: 
                network.append(torch.nn.ReLU())
        self.network = torch.nn.Sequential(*network) 
        self.out_dim = out_dim

    def forward(self, embedding_output, v_embedding_output):
    # def forward(self, embedding_output): 
        # embedding_last = embedding_output[:,-1,:]
        embedding_last = torch.mean(embedding_output, dim=1) 
        bsz, n1, n2 = v_embedding_output.shape
        vision_last = v_embedding_output.reshape(bsz, -1)

        input_vector = torch.cat([embedding_last, vision_last], dim=1)
        # input_vector = embedding_last 
        print(input_vector.shape)
        output = self.network(input_vector)
        return output 