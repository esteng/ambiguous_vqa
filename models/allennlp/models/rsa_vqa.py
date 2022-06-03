
import os 
import collections
import logging
from copy import deepcopy
from typing import Dict, List, Optional, Union
import pdb 
from pathlib import Path
from overrides import overrides
import torch
import numpy as np 

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model

from allennlp.modules.rsa_vqa.vqa_classifier import VQAClassifier
from allennlp.modules.rsa_vqa.speaker import BaseSpeakerModule
from allennlp.modules.rsa_vqa.listener import BaseListenerModule
from allennlp.data.fields.metadata_field import MetadataField
from allennlp.modules.vision.vision_language_encoder import CLIPLanguageEncoder, VisionLanguageEncoder, ViLTLanguageEncoder

from allennlp.nn.losses import (Loss, 
                                BCELoss, 
                                WeightedBCELoss, 
                                MultilabelCELoss, 
                                CELoss, 
                                CEAndBCELoss)

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
        vision_language_encoder: VisionLanguageEncoder,
        speaker_module: BaseSpeakerModule,
        listener_module: BaseListenerModule,
        num_listener_steps: int,
        copy_speaker_listener: bool,
        pooled_output_dim: int,
        beam_size: int = 5,
        dropout: float = 0.1,
        loss: Loss = CELoss(),
        vqa_loss_factor: float = 1.0,
        speaker_loss_factor: List[float] = [1.0],
        label_namespace: str = "answers",
        keep_tokens: bool = False,
        meaning_vector_source: str = "listener", 
    ) -> None:
        super().__init__(vocab)

        # self.debug_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.consistency_wrong_map: Dict[str, int] = collections.Counter()
        from allennlp.training.metrics import F1MultiLabelMeasure
        self.loss_fxn = loss
        # if isinstance(self.loss_fxn, BCELoss) or isinstance(self.loss_fxn, WeightedBCELoss) \
        #     or isinstance(self.loss_fxn, MultilabelCELoss): 
        #     self.f1_metric = BCEF1MultiLabelMeasure() 
        # else:
        self.f1_metric = F1MultiLabelMeasure(average="micro")

        from allennlp.training.metrics.vqa import VqaMeasure
        self.vqa_metric = VqaMeasure()

        from allennlp.training.metrics import BLEU
        exclude_indices = set(speaker_module._exclude_indices) | set([0])

        self.beam_size = beam_size
        speaker_module.decoder._beam_search.beam_size = beam_size 
        if copy_speaker_listener:
            speaker_modules = [deepcopy(speaker_module) for i in range(num_listener_steps)]
            listener_modules = [deepcopy(listener_module) for i in range(num_listener_steps)]
        else:
            speaker_modules = [speaker_module for i in range(num_listener_steps)]
            listener_modules = [listener_module for i in range(num_listener_steps)]

        self.acc_metrics = [BLEU(exclude_indices=exclude_indices) for i in range(len(speaker_modules))]

        self.vision_language_encoder = vision_language_encoder

        self.keep_tokens = keep_tokens
        if keep_tokens:
            self.encoded_token_projection = torch.nn.Linear(self.vision_language_encoder.encoder.hidden_size1, pooled_output_dim)

        self.pooled_output_projection = torch.nn.Linear(pooled_output_dim, speaker_module.encoder_in_dim)

        self.speaker_modules = torch.nn.ModuleList(speaker_modules)
        self.listener_modules = torch.nn.ModuleList(listener_modules)
        self.copy_speaker_listener = copy_speaker_listener 
        self.num_listener_steps = len(speaker_modules)

        # num_labels = vocab.get_vocab_size(label_namespace)
        # self.num_labels = num_labels
        self.num_labels = len(self.vision_language_encoder.model.config.label2id)

        self.label_namespace = label_namespace

        self.classifier = VQAClassifier(speaker_module.encoder_hidden_dim, self.num_labels)
        self.dropout = torch.nn.Dropout(dropout)

        self.vqa_loss_factor = vqa_loss_factor
        if type(speaker_loss_factor) == int:
            speaker_loss_factor = [speaker_loss_factor for i in range(len(speaker_modules))] 
        self.speaker_loss_factor = speaker_loss_factor
        
        self.meaning_vector_source = meaning_vector_source

    def _cache_meaning_vectors(self, meaning_vectors, precompute_metadata):
        checkpoint_dir = os.environ['CHECKPOINT_DIR']


        for layer in range(len(meaning_vectors)):
            for i in range(len(meaning_vectors[layer])):
                metadata = precompute_metadata[i]
                out_dir = Path(metadata['save_dir'])
                out_dir.mkdir(exist_ok=True, parents=True)
                checkpoint_file = out_dir.joinpath("checkpoint_info.txt")
                # save checkpoint info to make organization easier later 
                if not checkpoint_file.exists():
                    with open(checkpoint_file, 'w') as f:
                        f.write(str(checkpoint_dir))
                filename = out_dir.joinpath(f"{metadata['image_id']}_{metadata['question_id']}_{layer}.pt")
                if filename.exists():
                    continue
                else:
                    torch.save(meaning_vectors[layer][i], filename)


    @overrides
    def forward(
        self,  # type: ignore
        box_features: torch.Tensor,
        box_coordinates: torch.Tensor,
        question: TextFieldTensors,
        question_id: MetadataField,
        question_input: torch.Tensor = None,
        labels: Optional[torch.Tensor] = None,
        label_weights: Optional[torch.Tensor] = None,
        labels_from_pretrained: Optional[torch.Tensor] = None,
        labels_for_metric: Optional[torch.Tensor] = None,
        weights_for_metric: Optional[torch.Tensor] = None,
        debug_tokens: Optional[MetadataField] = None,
        debug_answer: Optional[MetadataField] = None,
        debug_images: Optional[MetadataField] = None,
        meaning_vectors_input: Optional[List[torch.Tensor]] = None,
        pooled_output: Optional[torch.Tensor] = None,
        vilt_data: Optional[Dict] = None,
        precompute_metadata: Optional[List[Dict]] = None,
    ) -> Dict[str, torch.Tensor]:

        # only run vision-language encoder if the reps haven't been pre-computed  
        if meaning_vectors_input is None and pooled_output is None:
            if isinstance(self.vision_language_encoder, CLIPLanguageEncoder) or isinstance(self.vision_language_encoder, ViLTLanguageEncoder):
                # TODO (elias) if we want to train full model, remove this
                with torch.no_grad() :
                    if vilt_data is not None:
                        pooled_output, __ = \
                            self.vision_language_encoder.supposed_to_be_fast_forward(vilt_data['input_ids'],
                                                        vilt_data['token_type_ids'],
                                                        vilt_data['attention_mask'],
                                                        vilt_data['pixel_values'],
                                                        vilt_data['pixel_mask'])
                    else:
                        pooled_output, __ = \
                            self.vision_language_encoder(debug_tokens,
                                                         debug_images)

            else: 
                pooled_output, sequence_output_t = self.vision_language_encoder(
                                                        box_features=box_features,
                                                        box_coordinates=box_coordinates,
                                                        question=question,
                                                        question_input=question_input)

            pooled_output = self.pooled_output_projection(pooled_output)
            if self.keep_tokens:
                encoded_tokens = self.encoded_token_projection(sequence_output_t)
                listener_output = torch.cat([pooled_output.unsqueeze(1), encoded_tokens], dim=1)
            else:
                listener_output = pooled_output

        elif pooled_output is not None:
            listener_output = pooled_output
        else:
            listener_output = None
            # question_input = None


        if meaning_vectors_input is None:
            meaning_vectors_input = [None for i in range(self.num_listener_steps)]
        else:
            listener_output = None
            # question_input = None

        speaker_losses = []
        meaning_vectors_output = []  
        prev_listener_outputs = [] 
        speaker_utterances = [[] for i in range(len(self.speaker_modules))]
        for i in range(self.num_listener_steps): 
            if listener_output is None:
                bsz = meaning_vectors_input[i].shape[0]
            else:
                bsz = listener_output.shape[0]

            if self.meaning_vector_source == "listener" and meaning_vectors_input[i] is None: 
                # meaning vector is set to listener output 
                meaning_vectors_output.append(listener_output)
                # run speaker module as normal, no modified output 
                speaker_output = self.speaker_modules[i](fused_representation=listener_output, 
                                                        gold_utterance=question_input)

                if meaning_vectors_input[0] is not None and i > 0 and listener_output.requires_grad:
                    # we're doing the min_gen procedure 
                    speaker_output['encoder_output']['encoder_outputs'].requires_grad = True

            elif self.meaning_vector_source == "listener" and meaning_vectors_input[i] is not None:
                # run speaker module with the modified listener output 
                listener_output = meaning_vectors_input[i].squeeze(0)
                # listener output is the meaning vector 
                meaning_vectors_output.append(listener_output)
                speaker_output = self.speaker_modules[i](fused_representation=listener_output,
                                                        gold_utterance=question_input,
                                                        speaker_encoder_output=meaning_vectors_input[i],
                                                        do_detach=True)
            else:
                # using speaker encoder output as meaning vector
                speaker_output = self.speaker_modules[i](fused_representation=listener_output,
                                                        gold_utterance=question_input,
                                                        speaker_encoder_output=meaning_vectors_input[i])



            speaker_loss = speaker_output['loss']
            speaker_losses.append(speaker_loss) 
            
            if question_input is not None: 
                speaker_predictions = speaker_output['predictions'].contiguous() 
                # get index of EOS 
                prediction_mask = torch.zeros_like(speaker_predictions)
                eos_prediction_indices = speaker_predictions == self.speaker_modules[i]._end_index 
                prediction_mask[eos_prediction_indices] = 1 
                prediction_mask = torch.cumsum(prediction_mask, dim=1).clamp(min=0, max=1)
                # keep EOS in
                prediction_mask[eos_prediction_indices] = 0
                prediction_mask = 1 - prediction_mask
                # mask out everything that comes after EOS 
                speaker_predictions *= prediction_mask 
                try:
                    self.acc_metrics[i](speaker_predictions,
                                        question_input['tokens']['tokens'][:,1:].contiguous())
                except ValueError:
                    self.acc_metrics[i](speaker_predictions[:, 0, :],
                                        question_input['tokens']['tokens'][:,1:].contiguous())

                if not self.training:
                    pred = self.speaker_modules[i].make_output_human_readable(speaker_output)['predicted_tokens']
                    speaker_utterances[i].append(pred)

            else:
                if not self.training:
                    pred = self.speaker_modules[i].make_output_human_readable(speaker_output)['predicted_tokens']
                    speaker_utterances[i].append(pred)

            encoded_by_speaker = speaker_output['encoder_output']['encoder_outputs']

            if self.meaning_vector_source == "speaker": 
                meaning_vectors_output.append(encoded_by_speaker)

            if not self.training: 
                if meaning_vectors_input[i] is None: 
                    beam_size = self.speaker_modules[i].decoder._beam_search.beam_size
                else:
                    beam_size = 1 
                    encoded_by_speaker = encoded_by_speaker[0, :]
                try:
                    encoded_by_speaker = encoded_by_speaker.reshape((bsz, beam_size, -1)) 
                except RuntimeError:
                    encoded_by_speaker = encoded_by_speaker.reshape((bsz, 1, -1)) 

            listener_mask = torch.ones_like(encoded_by_speaker)[:,:,0]
            listener_output = self.listener_modules[i](encoded_by_speaker,
                                                       listener_mask)['output'] 
            prev_listener_outputs.append(listener_output)

        logits = self.classifier(listener_output) 
        probs = torch.softmax(logits, dim=-1)

        predicted_labels = torch.argmax(logits, dim=1)

        outputs = {"logits": logits, "probs": probs, "speaker_loss": speaker_losses, 
                   "meaning_vectors_output": meaning_vectors_output, "speaker_utterances": speaker_utterances,
                   "predicted_labels": predicted_labels, "question_id": question_id}

        if (labels is not None or labels_from_pretrained is not None) and label_weights is not None:
            try:
                vqa_loss, weighted_labels, mask = self.loss_fxn(logits, labels_from_pretrained,  debug_answer, label_weights=label_weights)
            except:
                # mingen time, the batch size is the beam size
                vqa_loss, weighted_labels, mask = self.loss_fxn(logits[0,:].unsqueeze(0), labels_from_pretrained,  debug_answer, label_weights=label_weights)
            outputs['debug_answer'] = debug_answer
            if isinstance(self.loss_fxn, BCELoss) or isinstance(self.loss_fxn, WeightedBCELoss) or \
                isinstance(self.loss_fxn, CEAndBCELoss):

                binary_label_mask = mask 
                try:
                    self.f1_metric(logits, weighted_labels, binary_label_mask.bool())
                    self.vqa_metric(logits, labels_from_pretrained, weights_for_metric, do_interact=False, debug_tokens=debug_tokens, debug_answer=debug_answer) 
                except RuntimeError:
                    pass 
                except IndexError:
                    # mingen time, the batch size is the beam size 
                    self.f1_metric(logits[0,:].unsqueeze(0), weighted_labels, binary_label_mask.bool())
                    self.vqa_metric(logits[0,:].unsqueeze(0), labels_from_pretrained, weights_for_metric, do_interact=False, debug_tokens=debug_tokens, debug_answer=debug_answer) 

            elif isinstance(self.loss_fxn, MultilabelCELoss):  
                labels = labels.squeeze(1)
                labels_for_metric = torch.argmax(labels, dim=1)
                label_weights_for_f1_metric = torch.ones_like(labels_for_metric)
            else:
                labels_for_metric = labels 
                label_weights_for_metric = label_weights

            text_loss = 0.0
            for i, x in enumerate(speaker_losses): 
                text_loss += self.speaker_loss_factor[i] * x 
            big_loss = self.vqa_loss_factor * vqa_loss + text_loss

            outputs['loss'] =  big_loss
            outputs['vqa_loss'] = vqa_loss
            outputs['text_loss'] = text_loss

        return outputs


    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        result = self.f1_metric.get_metric(reset)
        result["vqa_score"] = self.vqa_metric.get_metric(reset)["score"]

        for i in range(len(self.acc_metrics)):
            result[f'speaker_bleu_{i}'] = self.acc_metrics[i].get_metric(reset)['BLEU'] 

        return result

    def eval_for_gen(self):
        self.eval()


@Model.register("precompute_vqa")
@Model.register("precompute_from_huggingface", constructor="from_huggingface_model_name")
class PrecomputeVQAModel(RSAVQAModel):
    """
    Keep everything the same so that we can use the same config, but most things here don't get used 
    """
    def __init__(
        self,
        vocab: Vocabulary,
        vision_language_encoder: VisionLanguageEncoder,
        speaker_module: BaseSpeakerModule,
        listener_module: BaseListenerModule,
        num_listener_steps: int,
        copy_speaker_listener: bool,
        pooled_output_dim: int,
        dropout: float = 0.1,
        losses: str = "ce",
        vqa_loss_factor: float = 1.0,
        speaker_loss_factor: float = 1.0,
        label_namespace: str = "answers",
        keep_tokens: bool = False,
    ) -> None:
        super().__init__(vocab,
                         vision_language_encoder,
                         speaker_module,
                         listener_module,
                         num_listener_steps,
                         copy_speaker_listener,
                         pooled_output_dim,
                         dropout,
                         losses,
                         vqa_loss_factor,
                         speaker_loss_factor,
                         label_namespace,
                         keep_tokens)


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
        label_indices: Optional[torch.Tensor] = None,
        debug_tokens: Optional[MetadataField] = None,
        debug_answer: Optional[MetadataField] = None,
        debug_images: Optional[MetadataField] = None,
        precompute_metadata: Optional[MetadataField] = None,
        speaker_encoder_outputs: Optional[List[torch.Tensor]] = None,
        pooled_output: Optional[torch.Tensor] = None,
        sequence_output: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size, _, feature_size = box_features.size()

        if isinstance(self.vision_language_encoder, CLIPLanguageEncoder) or isinstance(self.vision_language_encoder, ViLTLanguageEncoder):
            # TODO (elias) remove after debugging 
            with torch.no_grad() :
                pooled_output, sequence_output = self.vision_language_encoder(debug_tokens,
                                                                            debug_images)

        else: 
            pooled_output, sequence_output_t = self.vision_language_encoder(
                                                    box_features=box_features,
                                                    box_coordinates=box_coordinates,
                                                    question=question,
                                                    question_input=question_input)


        for i in range(pooled_output.shape[0]):
            metadata = precompute_metadata[i]
            out_dir = Path(metadata['save_dir'])
            out_dir.mkdir(exist_ok=True, parents=True)
            filename = out_dir.joinpath(f"{metadata['image_id']}_{metadata['question_id']}.pt")
            if filename.exists():
                continue
            else:
                torch.save(pooled_output[i], filename)

        return {"loss": torch.zeros(1, requires_grad=True)}
# 