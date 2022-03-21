import pdb
from platform import java_ver 
from typing import Tuple, Dict, Any, Optional
import torch
from einops import rearrange
import numpy 
from copy import deepcopy

from allennlp.data import Vocabulary
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.common.registrable import Registrable
from allennlp.modules.seq2seq_encoders.prenorm_transformer_encoder import MisoTransformerEncoder
from allennlp.modules.transformer_decoder.prenorm_transformer_decoder import MisoTransformerDecoder
from allennlp_models.generation.modules.decoder_nets import StackedSelfAttentionDecoderNet, DecoderNet
from allennlp_models.generation.modules.seq_decoders.seq_decoder import SeqDecoder
from allennlp.data.fields.text_field import TextFieldTensors
# from allennlp_models

PAD_SYMBOL = "@@UNKNOWN@@"
CLS_SYMBOL = "[CLS]"
SEP_SYMBOL = "[SEP]"
class BaseSpeakerModule(torch.nn.Module, Registrable):
# class BaseSpeakerModule(ComposedSeq2Seq):
    def __init__(self, 
                 vocab: Vocabulary,
                 target_namespace: str, 
                 encoder: Any, 
                 decoder: SeqDecoder,
                 dropout: float = 0.1): 
        super(BaseSpeakerModule, self).__init__()
        self.vocab = vocab
        self._target_namespace = target_namespace
        self.encoder = encoder 
        self.decoder = decoder 
        self.dropout = torch.nn.Dropout(dropout)
        self.activation = torch.nn.ReLU()
        self.loss = torch.nn.CrossEntropyLoss()

        self._start_index = self.vocab.get_token_index(START_SYMBOL, self._target_namespace)
        self._end_index = self.vocab.get_token_index(END_SYMBOL, self._target_namespace)
        self._pad_index = self.vocab.get_token_index(PAD_SYMBOL, self._target_namespace)
        self._cls_index = self.vocab.get_token_index(CLS_SYMBOL, self._target_namespace)
        self._sep_index = self.vocab.get_token_index(SEP_SYMBOL, self._target_namespace)
        self._exclude_indices = [self._start_index, self._end_index, self._pad_index, self._cls_index, self._sep_index]

@BaseSpeakerModule.register("prenorm_speaker")
class PrenormSpeakerModule(BaseSpeakerModule):
    def __init__(self, 
                 vocab: Vocabulary,
                 target_namespace: str,
                 encoder: MisoTransformerEncoder,
                 decoder: SeqDecoder,
                 dropout: float = 0.1): 
        super(PrenormSpeakerModule, self).__init__(vocab=vocab,
                                                   target_namespace=target_namespace,
                                                   encoder=encoder, 
                                                   decoder=decoder,
                                                   dropout=dropout)

        self.encoded_vec_for_backprop = torch.nn.parameter.Parameter(torch.zeros(1,1, encoder.hidden_size))

    def forward(self, 
                fused_representation: torch.Tensor,
                gold_utterance: torch.Tensor=None, 
                gold_utterance_mask: torch.Tensor=None,
                speaker_encoder_output: torch.Tensor=None): 
        if gold_utterance is not None: 
            return self._training_forward(target_tokens=gold_utterance,
                                          source_memory=fused_representation,
                                          target_mask=gold_utterance_mask)
        else:
            return self._test_forward(source_memory=fused_representation, 
                                      speaker_encoder_output=speaker_encoder_output)

    def _encode(self, embedded_input: torch.Tensor, source_mask: torch.Tensor): 
        encoder_outputs = self.encoder(embedded_input, source_mask)
        return {"source_mask": source_mask, "encoder_outputs": encoder_outputs}

    def _training_forward(self, 
                          target_tokens: TextFieldTensors,
                          source_memory: torch.Tensor,
                          source_mask: torch.Tensor = None,
                          target_mask: torch.Tensor = None): 
        # run forward and return loss 
        embedded = source_memory.unsqueeze(1)
        source_mask = torch.ones_like(embedded)[:,:,0].bool()
        encoded = self._encode(embedded, source_mask) 

        if not self.training:
            # detach during validation so it doesn't get copied 
            encoded_for_ret = {}
            for k in encoded.keys():
                encoded_for_ret[k] = encoded[k].detach().clone()
        else:
            encoded_for_ret = encoded
        decoded = self.decoder(encoded, target_tokens)

        top_k = torch.argmax(decoded['logits'], dim=2)
        loss = decoded['loss']

        return {"encoder_output": encoded_for_ret, "output": decoded['logits'], "predictions": top_k, "loss": loss}

    def _test_forward(self,
                      source_memory: torch.Tensor,
                      speaker_encoder_output: torch.Tensor=None):
        if speaker_encoder_output is None: 
            embedded = source_memory.unsqueeze(1)
            source_mask = torch.ones_like(embedded)[:,:,0].bool()
            encoded = self._encode(embedded, source_mask) 
            decoded = self.decoder(encoded) 
            loss = torch.nan
            return {"encoder_output": encoded, "predictions": decoded['predictions'], "loss": loss}
        else:
            # if it's not the first iteration, pass previous iteration's output in here 
            # intervene, replace encoded
            prev_speaker_output = speaker_encoder_output[0]
            if len(prev_speaker_output.shape) == 2:
                source_mask = torch.ones_like(prev_speaker_output.detach()[:,0].unsqueeze(0).bool())
            else:
                source_mask = torch.ones_like(prev_speaker_output.detach()[:,:,0].unsqueeze(0).bool())
            # try:
            #     print(f"in test forward: {prev_speaker_output[0,0,0:10]}") 
            # except:
            #     print(f"in test forward: {prev_speaker_output[0,0:10]}") 

            encoded = {'encoder_outputs': prev_speaker_output,
                                 'source_mask': source_mask}

            # don't decode for now, this is causing the issues because Beam search is non-diff
            encoded_for_decode = {'encoder_outputs': prev_speaker_output.detach().clone(),
                                 'source_mask': source_mask.detach().clone()}
            decoded = self.decoder(encoded_for_decode)
            loss = torch.nan
            return {"encoder_output": encoded, "predictions": decoded['predictions'], "loss": loss}

    def _prepare_output_projections(
        self, last_predictions: torch.Tensor, state: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Decode current state and last prediction to produce produce projections
        into the target space, which can then be used to get probabilities of
        each target token for the next step.
        """
        if ("speaker_outputs" in state and 
            len(state['speaker_outputs']) > 0 and 
            state['speaker_outputs'][0] is not None):
            # shape: (group_size, max_input_sequence_length, encoder_output_dim)
            encoder_outputs = state["encoder_outputs"].detach().clone()
        else:
            # shape: (group_size, max_input_sequence_length, encoder_output_dim)
            encoder_outputs = state["encoder_outputs"]

        # shape: (group_size, max_input_sequence_length)
        source_mask = state["source_mask"]

        # shape: (group_size, steps_count, decoder_output_dim)
        previous_steps_predictions = state.get("previous_steps_predictions")

        # shape: (batch_size, 1, target_embedding_dim)
        last_predictions_embeddings = self.target_embedder(last_predictions).unsqueeze(1)

        if previous_steps_predictions is None or previous_steps_predictions.shape[-1] == 0:
            # There is no previous steps, except for start vectors in `last_predictions`
            # shape: (group_size, 1, target_embedding_dim)
            previous_steps_predictions = last_predictions_embeddings
        else:
            # shape: (group_size, steps_count, target_embedding_dim)
            previous_steps_predictions = torch.cat(
                [previous_steps_predictions, last_predictions_embeddings], 1
            )

        decoded = self.decoder(inputs=state["previous_steps_predictions"],
                               source_memory_bank=encoder_outputs,
                               source_mask=source_mask,
                               target_mask=None)

        decoded = self.activation(self.dropout(decoded['outputs']))
        output = self.output_layer(decoded)

        state["previous_steps_predictions"] = previous_steps_predictions

        # Update state with new decoder state, override previous state

        # shape: (group_size, num_classes)

        return output, state

    def make_output_human_readable(self, output_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Finalize predictions.
        This method overrides `Model.make_output_human_readable`, which gets called after `Model.forward`, at test
        time, to finalize predictions. The logic for the decoder part of the encoder-decoder lives
        within the `forward` method.
        This method trims the output predictions to the first end symbol, replaces indices with
        corresponding tokens, and adds a field called `predicted_tokens` to the `output_dict`.
        """
        predicted_indices = output_dict["predictions"]
        if not isinstance(predicted_indices, numpy.ndarray):
            predicted_indices = predicted_indices.detach().cpu().numpy()
        all_predicted_tokens = []
        for top_k_predictions in predicted_indices:
            # Beam search gives us the top k results for each source sentence in the batch
            # we want top-k results.
            if len(top_k_predictions.shape) == 1:
                top_k_predictions = [top_k_predictions]

            batch_predicted_tokens = []
            for indices in top_k_predictions:
                indices = list(indices)
                # Collect indices till the first end_symbol
                if self._end_index in indices:
                    indices = indices[: indices.index(self._end_index)]
                predicted_tokens = [
                    self.vocab.get_token_from_index(x, namespace=self._target_namespace)
                    for x in indices
                ]
                batch_predicted_tokens.append(predicted_tokens)

            all_predicted_tokens.append(batch_predicted_tokens)
        output_dict["predicted_tokens"] = all_predicted_tokens
        return output_dict

@BaseSpeakerModule.register("simple_speaker")
class SimpleSpeakerModule(BaseSpeakerModule):
    def __init__(self, 
                 vocab: Vocabulary,
                 target_namespace: str,
                 encoder_in_dim: int, 
                 encoder_num_layers: int, 
                 encoder_hidden_dim: int,
                 encoder_dropout: float,
                 encoder_activation: str, 
                 decoder: SeqDecoder,
                 dropout: float = 0.1): 
        super(SimpleSpeakerModule, self).__init__(vocab=vocab,
                                                   target_namespace=target_namespace,
                                                   encoder=None, 
                                                   decoder=decoder,
                                                   dropout=dropout)
        layer0 = torch.nn.Linear(encoder_in_dim, encoder_hidden_dim)
        layers = [layer0] + [torch.nn.Linear(encoder_hidden_dim, encoder_hidden_dim) for i in range(encoder_num_layers - 1)]
        modules = []
        act = torch.nn.ReLU() if encoder_activation == "relu" else None
        if act is None:
            raise AssertionError("acivation must be ReLU")

        for l in layers: 
            modules.append(l)
            modules.append(torch.nn.Dropout(encoder_dropout))
            modules.append(deepcopy(act))

        
        self.encoder = torch.nn.Sequential(*layers) 

        self.encoded_vec_for_backprop = torch.nn.parameter.Parameter(torch.zeros(1,1, encoder_hidden_dim))

    def forward(self, 
                fused_representation: torch.Tensor,
                gold_utterance: torch.Tensor=None, 
                gold_utterance_mask: torch.Tensor=None,
                speaker_encoder_output: torch.Tensor=None): 
        if gold_utterance is not None: 
            return self._training_forward(target_tokens=gold_utterance,
                                          source_memory=fused_representation,
                                          target_mask=gold_utterance_mask)
        else:
            return self._test_forward(source_memory=fused_representation, 
                                      speaker_encoder_output=speaker_encoder_output)

    def _encode(self, embedded_input: torch.Tensor, source_mask: torch.Tensor): 
        encoder_outputs = self.encoder(embedded_input) 
        return {"source_mask": source_mask, "encoder_outputs": encoder_outputs}

    def _training_forward(self, 
                          target_tokens: TextFieldTensors,
                          source_memory: torch.Tensor,
                          source_mask: torch.Tensor = None,
                          target_mask: torch.Tensor = None): 
        # run forward and return loss 
        embedded = source_memory.unsqueeze(1)
        source_mask = torch.ones_like(embedded)[:,:,0].bool()
        encoded = self._encode(embedded, source_mask) 

        if not self.training:
            # detach during validation so it doesn't get copied 
            encoded_for_ret = {}
            for k in encoded.keys():
                encoded_for_ret[k] = encoded[k].detach().clone()
        else:
            encoded_for_ret = encoded
        decoded = self.decoder(encoded, target_tokens)

        top_k = torch.argmax(decoded['logits'], dim=2)
        loss = decoded['loss']

        return {"encoder_output": encoded_for_ret, "output": decoded['logits'], "predictions": top_k, "loss": loss}

    def _test_forward(self,
                      source_memory: torch.Tensor,
                      speaker_encoder_output: torch.Tensor=None):
        if speaker_encoder_output is None: 
            embedded = source_memory.unsqueeze(1)
            source_mask = torch.ones_like(embedded)[:,:,0].bool()
            encoded = self._encode(embedded, source_mask) 
            decoded = self.decoder(encoded) 
            loss = torch.nan
            return {"encoder_output": encoded, "predictions": decoded['predictions'], "loss": loss}
        else:
            # if it's not the first iteration, pass previous iteration's output in here 
            # intervene, replace encoded
            prev_speaker_output = speaker_encoder_output[0]
            if len(prev_speaker_output.shape) == 2:
                source_mask = torch.ones_like(prev_speaker_output.detach()[:,0].unsqueeze(0).bool())
            else:
                source_mask = torch.ones_like(prev_speaker_output.detach()[:,:,0].unsqueeze(0).bool())
            # try:
            #     print(f"in test forward: {prev_speaker_output[0,0,0:10]}") 
            # except:
            #     print(f"in test forward: {prev_speaker_output[0,0:10]}") 

            encoded = {'encoder_outputs': prev_speaker_output,
                                 'source_mask': source_mask}

            # don't decode for now, this is causing the issues because Beam search is non-diff
            encoded_for_decode = {'encoder_outputs': prev_speaker_output.detach().clone(),
                                 'source_mask': source_mask.detach().clone()}
            decoded = self.decoder(encoded_for_decode)
            loss = torch.nan
            return {"encoder_output": encoded, "predictions": decoded['predictions'], "loss": loss}

    def _prepare_output_projections(
        self, last_predictions: torch.Tensor, state: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Decode current state and last prediction to produce produce projections
        into the target space, which can then be used to get probabilities of
        each target token for the next step.
        """
        if ("speaker_outputs" in state and 
            len(state['speaker_outputs']) > 0 and 
            state['speaker_outputs'][0] is not None):
            # shape: (group_size, max_input_sequence_length, encoder_output_dim)
            encoder_outputs = state["encoder_outputs"].detach().clone()
        else:
            # shape: (group_size, max_input_sequence_length, encoder_output_dim)
            encoder_outputs = state["encoder_outputs"]

        # shape: (group_size, max_input_sequence_length)
        source_mask = state["source_mask"]

        # shape: (group_size, steps_count, decoder_output_dim)
        previous_steps_predictions = state.get("previous_steps_predictions")

        # shape: (batch_size, 1, target_embedding_dim)
        last_predictions_embeddings = self.target_embedder(last_predictions).unsqueeze(1)

        if previous_steps_predictions is None or previous_steps_predictions.shape[-1] == 0:
            # There is no previous steps, except for start vectors in `last_predictions`
            # shape: (group_size, 1, target_embedding_dim)
            previous_steps_predictions = last_predictions_embeddings
        else:
            # shape: (group_size, steps_count, target_embedding_dim)
            previous_steps_predictions = torch.cat(
                [previous_steps_predictions, last_predictions_embeddings], 1
            )

        decoded = self.decoder(inputs=state["previous_steps_predictions"],
                               source_memory_bank=encoder_outputs,
                               source_mask=source_mask,
                               target_mask=None)

        decoded = self.activation(self.dropout(decoded['outputs']))
        output = self.output_layer(decoded)

        state["previous_steps_predictions"] = previous_steps_predictions

        # Update state with new decoder state, override previous state

        # shape: (group_size, num_classes)

        return output, state

    def make_output_human_readable(self, output_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Finalize predictions.
        This method overrides `Model.make_output_human_readable`, which gets called after `Model.forward`, at test
        time, to finalize predictions. The logic for the decoder part of the encoder-decoder lives
        within the `forward` method.
        This method trims the output predictions to the first end symbol, replaces indices with
        corresponding tokens, and adds a field called `predicted_tokens` to the `output_dict`.
        """
        predicted_indices = output_dict["predictions"]
        if not isinstance(predicted_indices, numpy.ndarray):
            predicted_indices = predicted_indices.detach().cpu().numpy()
        all_predicted_tokens = []
        for top_k_predictions in predicted_indices:
            # Beam search gives us the top k results for each source sentence in the batch
            # we want top-k results.
            if len(top_k_predictions.shape) == 1:
                top_k_predictions = [top_k_predictions]

            batch_predicted_tokens = []
            for indices in top_k_predictions:
                indices = list(indices)
                # Collect indices till the first end_symbol
                if self._end_index in indices:
                    indices = indices[: indices.index(self._end_index)]
                predicted_tokens = [
                    self.vocab.get_token_from_index(x, namespace=self._target_namespace)
                    for x in indices
                ]
                batch_predicted_tokens.append(predicted_tokens)

            all_predicted_tokens.append(batch_predicted_tokens)
        output_dict["predicted_tokens"] = all_predicted_tokens
        return output_dict
