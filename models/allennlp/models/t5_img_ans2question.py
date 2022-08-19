
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

from allennlp.training.metrics import BLEU, ROUGE
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
        no_image_baseline: bool = False,
        no_answer_baseline: bool = False,
        no_vilt_train: bool = True,
        save_encoder_states: bool = False,
        save_encoder_states_args: Dict = None,
    ) -> None:
        super().__init__(vocab)

        self.beam_size = beam_size
        self.vision_language_encoder = vision_language_encoder

        self.save_encoder_states = save_encoder_states
        self.save_encoder_states_args = save_encoder_states_args

        self.t5_model_full = T5ForConditionalGeneration.from_pretrained(t5_model_name)
        self.t5_tokenizer = AutoTokenizer.from_pretrained(t5_model_name)
        # BLEU 1, 2, 3, 4, ROUGE-L, and CIDER
        exclude_tokens = [self.t5_tokenizer.pad_token, self.t5_tokenizer.eos_token, self.t5_tokenizer.unk_token]
        exclude_indices = set(self.t5_tokenizer.convert_tokens_to_ids(exclude_tokens))
        self.bleu_metrics = [BLEU(ngram_weights = (1, 0, 0, 0), exclude_indices = exclude_indices),
                            BLEU(ngram_weights = (0, 1, 0, 0), exclude_indices = exclude_indices),
                            BLEU(ngram_weights = (0, 0, 1, 0), exclude_indices = exclude_indices),
                            BLEU(ngram_weights = (0, 0, 0, 1), exclude_indices = exclude_indices)]
        self.rouge_metric = ROUGE(exclude_indices=exclude_indices)


        self.pooled_output_projection = torch.nn.Linear(pooled_output_dim, self.t5_model_full.model_dim)

        self.num_labels = len(self.vision_language_encoder.model.config.label2id)

        self.dropout = torch.nn.Dropout(dropout)
        self.no_image_baseline = no_image_baseline 
        self.no_answer_baseline = no_answer_baseline 
        self.no_vilt_train = no_vilt_train

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

        if self.no_vilt_train:
            with torch.no_grad():
                __, vilt_seq_output = \
                    self.vision_language_encoder(answers_as_text,
                                                debug_images,
                                                no_image_baseline = self.no_image_baseline,
                                                no_answer_baseline = self.no_answer_baseline)
        else:
            __, vilt_seq_output = \
                self.vision_language_encoder(answers_as_text,
                                            debug_images,
                                            no_image_baseline = self.no_image_baseline,
                                            no_answer_baseline = self.no_answer_baseline)

        seq_output = vilt_seq_output.float()
        seq_output = self.pooled_output_projection(seq_output)

        if debug_tokens is not None:
            labels = self.t5_tokenizer(debug_tokens, return_tensors='pt', padding=True).input_ids.to(seq_output.device)
        # if question_output is not None:
        # do training forward
        if self.training:
            outputs = self.t5_model_full(inputs_embeds = seq_output,
                                        labels = labels) 
            loss = outputs.loss
            if np.isnan(loss.item()):
                pdb.set_trace()
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
                    try:
                        pred_dict = self.t5_model_full.generate(inputs_embeds = seq_slice, 
                                                            num_beams=self.beam_size, 
                                                            force_words_ids = force_slice,
                                                            output_hidden_states=True,
                                                            return_dict_in_generate=True)
                    except ValueError:
                        # In the rare case that a word gets split into subwords and there are repeated subwords
                        # we will just ignore it    
                        pred_dict = self.t5_model_full.generate(inputs_embeds = seq_slice, 
                                                            num_beams=self.beam_size, 
                                                            force_words_ids = None,
                                                            output_hidden_states=True,
                                                            return_dict_in_generate=True)



                    pred_slice = pred_dict['sequences']
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
                pred_dict = self.t5_model_full.generate(inputs_embeds = seq_output, 
                                                num_beams=self.beam_size, 
                                                output_hidden_states=True,
                                                return_dict_in_generate=True)
                pred_toks = pred_dict['sequences']

            # save if needed 
            if self.save_encoder_states: 
                all_encoder_states = pred_dict['encoder_hidden_states']
                if self.save_encoder_states_args['vilt_only']:
                    encoder_states = vilt_seq_output 
                else:
                    encoder_states = all_encoder_states[self.save_encoder_states_args['layer']]
                # get just the answer embeddings 
                if self.save_encoder_states_args['just_answer']: 
                    # use tokenizer to tokenize answer 
                    tokenized_answers = [self.vision_language_encoder.tokenizer.tokenize(x) for x in answers_as_text]
                    # get the length of each tokenized answer 
                    answer_lens = [len(x) for x in tokenized_answers]
                    encoder_states = [encoder_states[i,0:l, :] for i, l in enumerate(answer_lens)]
                    # pooling needs to be changed since it's not a padded tensor anymore 
                    if self.save_encoder_states_args['pooling'] == "mean": 
                        encoder_states = [encoder_states[i].mean(dim=0) for i in range(encoder_states.shape[0])]
                    elif self.save_encoder_states_args['pooling'] == "max":
                        encoder_states = [encoder_states[i].max(dim=0) for i in range(encoder_states.shape[0])]
                    elif self.save_encoder_states_args['pooling'] == "none":
                        pass 
                    else:
                        raise ValueError(f"Invalid pooling method {self.save_encoder_states_args['pooling']}")
                else:
                    if self.save_encoder_states_args['pooling'] == "mean": 
                        encoder_states = encoder_states.mean(dim=1)
                    elif self.save_encoder_states_args['pooling'] == "max":
                        encoder_states = encoder_states.max(dim=1)[0]
                    elif self.save_encoder_states_args['pooling'] == "none":
                        pass 
                    else:
                        raise ValueError(f"Invalid pooling method {self.save_encoder_states_args['pooling']}")

                for i, qid in enumerate(question_id): 
                    save_path = self.save_encoder_states_args['path'] + f"/{qid}.pt"
                    torch.save(encoder_states[i], save_path)
            loss = torch.zeros(1,1).to(seq_output.device)
            
                
        speaker_utterances = []
        if debug_tokens is not None:
            for i, bleu_metric in enumerate(self.bleu_metrics):
                self.bleu_metrics[i](pred_toks,
                                     labels)
            self.rouge_metric(pred_toks, labels) 

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
        result = {}
        for i in range(len(self.bleu_metrics)):
            result.update(self.bleu_metrics[i].get_metric(reset, bleu_prefix=i+1))
        result.update(self.rouge_metric.get_metric(reset))
        return result

    def eval_for_gen(self):
        self.eval()

