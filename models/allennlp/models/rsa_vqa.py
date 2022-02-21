from cProfile import label
import collections
import logging
from copy import deepcopy
from typing import Dict, List, Optional, Union
import pdb 

from overrides import overrides
import torch
from transformers import AutoModel, AutoTokenizer

from allennlp.data import TextFieldTensors, Vocabulary, TokenIndexer
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder
from allennlp.modules.transformer import (
    TextEmbeddings,
    ImageFeatureEmbeddings,
    BiModalEncoder,
    TransformerPooler,
)
from allennlp.nn import util
from allennlp.common.params import Params
from allennlp.modules.rsa_vqa.speaker import BaseSpeakerModule
from allennlp.modules.rsa_vqa.listener import BaseListenerModule
from allennlp.data.fields.metadata_field import MetadataField

logger = logging.getLogger(__name__)

@Model.register("rsa_vqa")
@Model.register("rsa_vqa_from_huggingface", constructor="from_huggingface_model_name")
class RSAVQAModel(Model):
    """
    The model
    """
    def __init__(
        self,
        vocab: Vocabulary,
        text_embeddings: TextEmbeddings,
        image_embeddings: ImageFeatureEmbeddings,
        encoder: BiModalEncoder,
        speaker_modules: List[BaseSpeakerModule],
        listener_modules: List[BaseListenerModule],
        copy_speaker_listener: bool,
        pooled_output_dim: int,
        fusion_method: str = "sum",
        dropout: float = 0.1,
        vqa_loss_factor: float = 1.0,
        label_namespace: str = "answers",
        keep_tokens: bool = False,
        tune_bert: bool = False,
        tune_images: bool = False,
    ) -> None:
        super().__init__(vocab)

        # self.debug_tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.consistency_wrong_map: Dict[str, int] = collections.Counter()
        from allennlp.training.metrics import F1MultiLabelMeasure
        self.f1_metric = F1MultiLabelMeasure(average="micro")

        from allennlp.training.metrics.vqa import VqaMeasure
        self.vqa_metric = VqaMeasure()

        from allennlp.training.metrics import CategoricalAccuracy
        from allennlp.training.metrics import BLEU
        exclude_indices = set(speaker_modules[0]._exclude_indices)
        self.acc_metrics = [BLEU(exclude_indices=exclude_indices) for i in range(len(speaker_modules))]

        self.fusion_method = fusion_method

        self.embeddings = text_embeddings
        self.keep_tokens = keep_tokens
        self.tune_bert = tune_bert 
        self.tune_images = tune_images
        self.image_embeddings = image_embeddings
        self.encoder = encoder

        if keep_tokens:
            self.encoded_token_projection = torch.nn.Linear(encoder.hidden_size1, pooled_output_dim)

        self.t_pooler = TransformerPooler(encoder.hidden_size1, pooled_output_dim)
        self.v_pooler = TransformerPooler(encoder.hidden_size2, pooled_output_dim)

        self.speaker_modules = torch.nn.ModuleList(speaker_modules)
        self.listener_modules = torch.nn.ModuleList(listener_modules)
        self.copy_speaker_listener = copy_speaker_listener 
        self.num_listener_steps = len(speaker_modules)

        num_labels = vocab.get_vocab_size(label_namespace)
        self.num_labels = num_labels
        self.label_namespace = label_namespace

        self.classifier = torch.nn.Linear(self.encoder.hidden_size2, num_labels)
        self.dropout = torch.nn.Dropout(dropout)
        self.loss = torch.nn.CrossEntropyLoss()
        self.vqa_loss_factor = vqa_loss_factor

    @classmethod
    def from_huggingface_model_name(
        cls,
        vocab: Vocabulary,
        label_namespace: str,
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
        vqa_loss_factor: float = 1.0,
        num_listener_steps: int = 1,
        copy_speaker_listener: bool = True,
        tune_bert: bool = False,
        tune_images: bool = False,
        keep_tokens: bool = False,
        speaker_module: BaseSpeakerModule = None,
        listener_module: BaseListenerModule = None,
    ):
        transformer = AutoModel.from_pretrained(model_name)

        # TODO(mattg): This call to `transformer.embeddings` works with some transformers, but I'm
        # not sure it works for all of them, or what to do if it fails.
        # We should probably pull everything up until the instantiation of the image feature
        # embedding out into a central "transformers_util" module, or something, and just have a
        # method that pulls an initialized embedding layer out of a huggingface model.  One place
        # for this somewhat hacky code to live, instead of having to duplicate it in various models.
        text_embeddings = deepcopy(transformer.embeddings)

        image_embeddings = ImageFeatureEmbeddings(
            feature_dim=image_feature_dim,
            hidden_dim=image_hidden_size,
            dropout=image_hidden_dropout,
        )

        l0_encoder = BiModalEncoder.from_pretrained_module(
            pretrained_module=transformer,
            num_hidden_layers2=image_num_hidden_layers,
            hidden_size2=image_hidden_size,
            num_attention_heads2=image_num_attention_heads,
            combined_hidden_size=combined_hidden_size,
            combined_num_attention_heads=combined_num_attention_heads,
            intermediate_size2=image_intermediate_size,
            attention_dropout2=image_attention_dropout,
            hidden_dropout2=image_hidden_dropout,
            biattention_id1=text_biattention_id,
            biattention_id2=image_biattention_id,
            fixed_layer1=text_fixed_layer,
            fixed_layer2=image_fixed_layer,
        )
        speaker_stack = []
        listener_stack = []

        # speaker_params['vocab_size'] = text_embeddings.word_embeddings.num_embeddings
        # speaker_params['decoder']['target_embedder']['vocab'] = vocab
        # listener_params['vocab_size'] = num_labels

        if copy_speaker_listener: 
            for i in range(len(num_listener_steps)):
                # instantiate separate speakers and listeners for each layer 

                # speaker = BaseSpeakerModule.from_params(Params(speaker_params))
                speaker = deepcopy(speaker_module)
                listener = deepcopy(listener_module)
                speaker_stack.append(speaker)
                listener_stack.append(listener)

        else:
            # take advantage of the fact that * references the same object 
            speaker_stack = [speaker_module] * num_listener_steps
            listener_stack = [listener_module] * num_listener_steps

        return cls(
            vocab=vocab,
            label_namespace=label_namespace,
            text_embeddings=text_embeddings,
            image_embeddings=image_embeddings,
            encoder=l0_encoder,
            speaker_modules=speaker_stack,
            listener_modules=listener_stack,
            pooled_output_dim=pooled_output_dim,
            fusion_method=fusion_method,
            dropout=pooled_dropout,
            vqa_loss_factor=vqa_loss_factor,
            keep_tokens=keep_tokens,
            tune_bert=tune_bert, 
            tune_images=tune_images,
            copy_speaker_listener=copy_speaker_listener,
        )

    @overrides
    def forward(
        self,  # type: ignore
        box_features: torch.Tensor,
        box_coordinates: torch.Tensor,
        question: TextFieldTensors,
        question_input: torch.Tensor = None,
        # question_output: torch.Tensor = None,
        labels: Optional[torch.Tensor] = None,
        label_weights: Optional[torch.Tensor] = None,
        debug_tokens: Optional[MetadataField] = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size, _, feature_size = box_features.size()

        # TODO(mattg): have this make fewer assumptions.
        input_ids = question["tokens"]["token_ids"]
        token_type_ids = question["tokens"]["type_ids"]
        attention_mask = question["tokens"]["mask"]
        
        def get_embeddings(input_ids, token_type_ids, question_input):
            question_embedded_input = self.embeddings(input_ids, token_type_ids)
            num_tokens = question_embedded_input.size(1)
            # pdb.set_trace() 
            # if question_input is not None:
            #     # question_embedded_for_teacher_forcing = self.token_embeddings(question_input['tokens']['tokens'], question_input['tokens']['type_ids'])
            # else:
            #     question_embedded_for_teacher_forcing = None
            return question_embedded_input, num_tokens # , question_embedded_for_teacher_forcing

        # Get text embedding 
        if self.tune_bert:
            question_embedded_input, num_tokens = get_embeddings(input_ids, token_type_ids, question_input)
        else:
            with torch.no_grad():
                question_embedded_input, num_tokens = get_embeddings(input_ids, token_type_ids, question_input)

        # get image region embedding 
        if self.tune_images: 
            v_embedding_output = self.image_embeddings(box_features, box_coordinates)
        else:
            with torch.no_grad():
                # (batch_size, num_boxes, image_embedding_dim)
                v_embedding_output = self.image_embeddings(box_features, box_coordinates)

        # All batch instances will always have the same number of images and boxes, so no masking
        # is necessary, and this is just a tensor of ones.
        image_attention_mask = torch.ones_like(box_coordinates[:, :, 0])

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of
        # causal attention used in OpenAI GPT, we just need to prepare the
        # broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2).float().log()
        extended_image_attention_mask = image_attention_mask.unsqueeze(1).unsqueeze(2).float().log()

        extended_co_attention_mask = torch.zeros(
            batch_size,
            feature_size,
            num_tokens,
            dtype=extended_image_attention_mask.dtype,
        )


        encoded_layers_t, encoded_layers_v = self.encoder(
            question_embedded_input,
            v_embedding_output,
            extended_attention_mask,
            extended_image_attention_mask,
            extended_co_attention_mask,
        )

        # Pooling and fusing into one single representation 
        
        sequence_output_t = encoded_layers_t[:, :, :, -1]
        sequence_output_v = encoded_layers_v[:, :, :, -1]

        pooled_output_t = self.t_pooler(sequence_output_t)
        pooled_output_v = self.v_pooler(sequence_output_v)

        if self.fusion_method == "sum":
            pooled_output = self.dropout(pooled_output_t + pooled_output_v)
        elif self.fusion_method == "mul":
            pooled_output = self.dropout(pooled_output_t * pooled_output_v)
        else:
            raise ValueError(f"Fusion method '{self.fusion_method}' not supported")

        if self.keep_tokens:
            encoded_tokens = self.encoded_token_projection(sequence_output_t)
            listener_output = torch.cat([pooled_output.unsqueeze(1), encoded_tokens], dim=1)
        else:
            listener_output = pooled_output
        speaker_losses = []

        # TODO (elias): embed question input with a token embedder!
        for i in range(self.num_listener_steps): 
            bsz = listener_output.shape[0]
            speaker_output = self.speaker_modules[i](fused_representation=listener_output,
                                                     gold_utterance=question_input)
                                                    #  gold_utterance_output=question_output)
            speaker_loss = speaker_output['loss']
            speaker_losses.append(speaker_loss) 

            self.acc_metrics[i](speaker_output['predictions'].contiguous(), 
                                question_input['tokens']['tokens'][:,1:].contiguous())
            # pred = self.debug_tokenizer.batch_decode(speaker_output['top_k'].unsqueeze(1).reshape((bsz, -1))) 
            # true = self.debug_tokenizer.batch_decode(question_input['tokens']['tokens'], vocabulary=self.vocab)
            gold_predictions = {"predictions": question_input['tokens']['tokens'][:,1:]}
            pred = self.speaker_modules[i].make_output_human_readable(speaker_output)['predicted_tokens']
            true = self.speaker_modules[i].make_output_human_readable(gold_predictions)['predicted_tokens']
            
            logger.info("")
            logger.info(f"pred: {pred[0]}")
            logger.info(f"true: {true[0]}")
            logger.info(f"pred: {pred[1]}")
            logger.info(f"true: {true[1]}")

            encoded_by_speaker = speaker_output['encoder_output']['encoder_outputs']
            listener_mask = torch.ones_like(encoded_by_speaker)[:,:,0]
            listener_output = self.listener_modules[i](encoded_by_speaker,
                                                       listener_mask) 

        logits = self.classifier(listener_output['output']) 
        probs = torch.softmax(logits, dim=-1)

        outputs = {"logits": logits, "probs": probs, "speaker_loss": speaker_losses}
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

            loss = (
                torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, weighted_labels, weight=binary_label_mask, reduction="sum"
                )
                / batch_size
            )
 
            self.f1_metric(logits, weighted_labels, binary_label_mask.bool())
            self.vqa_metric(logits, labels, label_weights)
            # logger.info(f"loss: {loss.item()}")
            # speaker_loss_sum = torch.sum(torch.Tensor(speaker_losses))
            # logger.info(f"speaker loss: {speaker_loss_sum.item()}")
            losses = [self.vqa_loss_factor * loss] + speaker_losses
            # losses = speaker_losses
            big_loss = 0.0
            for loss in losses:
                big_loss += loss
            outputs['loss'] =  big_loss
        return outputs

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        result = self.f1_metric.get_metric(reset)
        result["vqa_score"] = self.vqa_metric.get_metric(reset)["score"]
        for i in range(len(self.acc_metrics)):
            result[f'speaker_bleu_{i}'] = self.acc_metrics[i].get_metric(reset)['BLEU'] 

        return result
