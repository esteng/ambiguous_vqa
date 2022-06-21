
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
from allennlp.data.fields.metadata_field import MetadataField
from allennlp.modules.rsa_vqa.speaker import BaseSpeakerModule
from allennlp.modules.vision.vision_language_encoder import CLIPLanguageEncoder, VisionLanguageEncoder, ViLTLanguageEncoder
# from allennlp.modules.transformer.t5 import T5DecoderStack

from allennlp.nn.losses import (Loss, 
                                BCELoss, 
                                WeightedBCELoss, 
                                MultilabelCELoss, 
                                CELoss, 
                                CEAndBCELoss)

from transformers import AutoModelWithLMHead, T5ForConditionalGeneration, AutoTokenizer

logger = logging.getLogger(__name__)

@Model.register("t5_img_ans_2_question")
@Model.register("t5_img_ans_2_question_from_huggingface", constructor="from_huggingface_model_name")
class T5ImageAnswer2QuestionModel(Model):
    """
    The model to go from Image + Answer to Question 
    """
    def __init__(
        self,
        vocab: Vocabulary,
        vision_language_encoder: VisionLanguageEncoder,
        t5_model_name: str,
        pooled_output_dim: int,
        beam_size: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(vocab)

        from allennlp.training.metrics import BLEU
        self.acc_metrics = BLEU()
        self.beam_size = beam_size
        self.vision_language_encoder = vision_language_encoder

        self.t5_model_full = T5ForConditionalGeneration.from_pretrained(t5_model_name)
        self.t5_tokenizer = AutoTokenizer.from_pretrained(t5_model_name)
        # for now, keep the encoder and re-encode vilt output 
        # might be wasteful but easier for implementation 
        # pdb.set_trace()
        # copy decoder and lm head 
        # self.t5_decoder = deepcopy(t5_model_full.decoder)
        # self.lm_head = deepcopy(t5_model_full.lm_head)
        # kill the rest of the model 
        # del(t5_model_full)

        # self.question_output_module = question_output_module

        self.pooled_output_projection = torch.nn.Linear(pooled_output_dim, self.t5_model_full.model_dim)

        self.num_labels = len(self.vision_language_encoder.model.config.label2id)

        self.dropout = torch.nn.Dropout(dropout)

    @overrides
    def forward(
        self,  # type: ignore
        question: TextFieldTensors,
        answers_as_text: MetadataField,
        answer_counts: torch.Tensor,
        question_id: MetadataField,
        question_output: Optional[torch.Tensor] = None,
        debug_tokens: Optional[MetadataField] = None,
        debug_answer: Optional[MetadataField] = None,
        debug_images: Optional[MetadataField] = None,
        force_toks: Optional[MetadataField] = None,
    ) -> Dict[str, torch.Tensor]:

        with torch.no_grad():
            __, seq_output = \
                self.vision_language_encoder(answers_as_text,
                                            debug_images)
        seq_output = seq_output.float()
        seq_output = self.pooled_output_projection(seq_output)

        if debug_tokens is not None:
            labels = self.t5_tokenizer(debug_tokens, return_tensors='pt', padding=True).input_ids.to(seq_output.device)
        # if question_output is not None:
        # do training forward
        if self.training:
            outputs = self.t5_model_full(inputs_embeds = seq_output,
                                        labels = labels) 
            loss = outputs.loss
            logits = outputs['logits']
            pred_toks = torch.argmax(logits, dim=-1)
        else:
            # TODO (elias): try using force_word_ids in generate to force generation of 
            # question target. First need a way to extract question target, good heuristic
            # may be to extract NPs, e.g. "what are the letters on the sign" -> "the letters, the sign"
            if force_toks is not None:
                force_words_ids = []
                for tok_seq in force_toks:
                    if len(tok_seq) > 0:
                        tokenized = self.t5_tokenizer(tok_seq, padding=False, add_special_tokens=False)
                        # pdb.set_trace()
                        input_ids = [[y]  for x in tokenized['input_ids'] for y in x]
                        # pdb.set_trace()
                        force_words_ids.append(input_ids)
                    else:
                        force_words_ids.append(None)

            else:
                force_words_ids = None
            if force_words_ids is not None and len(force_words_ids) == 0:
                force_words_ids = None
            if force_words_ids is not None: 
                pred_toks, pred_lens = [], []
                for i in range(seq_output.shape[0]): 
                    seq_slice = seq_output[i,:,:].unsqueeze(0)
                    try:
                        if force_words_ids[i] is None:
                            force_slice = None
                        else:
                            force_slice = [force_words_ids[i]]
                    except IndexError:
                        pdb.set_trace()
                    pred_slice = self.t5_model_full.generate(inputs_embeds = seq_slice, 
                                                            num_beams=self.beam_size, 
                                                            force_words_ids = force_slice)

                    # pdb.set_trace() 
                    pred_lens.append(pred_slice.shape[1])
                    pred_toks.append(pred_slice) 
                max_len = max(pred_lens)
                for i, pred in enumerate(pred_toks): 
                    curr_len = pred.shape[1]
                    pred = torch.nn.functional.pad(pred, (0, max_len - curr_len), value=self.t5_tokenizer.pad_token_id)
                    pred_toks[i] = pred
                pred_toks = torch.cat(pred_toks, dim=0).squeeze(1)
                # pdb.set_trace() 

            else:
                pred_toks = self.t5_model_full.generate(inputs_embeds = seq_output, 
                                                num_beams=self.beam_size) 

            # pdb.set_trace()
            loss = torch.zeros(1,1).to(seq_output.device)
            
            # loss = np.inf 
        # else:
        #     # do test forward 
        #     pdb.set_trace()
        #     outputs = self.t5_decoder.generate(input_embeds = seq_output)
        #     loss = -np.inf
        #     logits = outputs['logits']
        #     pred_toks = torch.argmax(logits, dim=-1)


        # generated_output = self.question_output_module(fused_representation = pooled_output,
                                                    #   gold_utterance = question_output)
                
        speaker_utterances = []
        # question_predictions = generated_output['predictions'].contiguous() 
        if debug_tokens is not None:
            self.acc_metrics(pred_toks,
                             labels) 

        if not self.training:
            # pdb.set_trace()
            question_predictions = self.t5_tokenizer.batch_decode(pred_toks)
            # pdb.set_trace()
            # pred = self.question_output_module.make_output_human_readable(generated_output)['predicted_tokens']
            speaker_utterances.append(question_predictions)

        # loss = generated_output['loss']
        outputs = {"question_id": question_id, 
                   "loss": loss, 
                   "speaker_utterances": speaker_utterances}
        return outputs

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        result = self.acc_metrics.get_metric(reset)
        return result

    def eval_for_gen(self):
        self.eval()

