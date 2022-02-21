import pdb 
import torch
from einops import rearrange

from allennlp.common.registrable import Registrable
from allennlp.modules.seq2seq_encoders.prenorm_transformer_encoder import MisoTransformerEncoder
from allennlp.modules.transformer_decoder.prenorm_transformer_decoder import MisoTransformerDecoder

class BaseSpeakerModule(torch.nn.Module, Registrable):
    def __init__(self, 
                 encoder: MisoTransformerEncoder,
                 decoder: MisoTransformerDecoder,
                 vocab_size: int,
                 dropout: float = 0.1): 
        super(BaseSpeakerModule, self).__init__()
        self.encoder = encoder 
        self.decoder = decoder 
        self.dropout = torch.nn.Dropout(dropout)
        self.activation = torch.nn.ReLU()
        out_size = [v for k,v in self.decoder.layers[-1]._modules.items() if "linear" in k][-1].out_features
        self.output_layer = torch.nn.Linear(out_size, vocab_size)
        self.loss = torch.nn.CrossEntropyLoss()

    


@BaseSpeakerModule.register("prenorm_speaker")
class PrenormSpeakerModule(BaseSpeakerModule):
    def __init__(self, 
                 encoder: MisoTransformerEncoder,
                 decoder: MisoTransformerDecoder,
                 vocab_size: int,
                 dropout: float = 0.1): 
        super(PrenormSpeakerModule, self).__init__(encoder=encoder, 
                                                   decoder=decoder,
                                                   vocab_size=vocab_size,
                                                   dropout=dropout)

    def forward(self, 
                fused_representation: torch.Tensor,
                gold_utterance_input: torch.Tensor=None, 
                gold_utterance_output: torch.Tensor=None,
                gold_utterance_mask: torch.Tensor=None): 
        if gold_utterance_input is not None: 
            return self._training_forward(decoder_inputs=gold_utterance_input,
                                          source_memory=fused_representation,
                                          decoder_targets=gold_utterance_output,
                                          target_mask=gold_utterance_mask)
        else:
            return self._test_forward(fused_representation)

    def _training_forward(self, 
                          decoder_inputs: torch.Tensor,
                          source_memory: torch.Tensor,
                          decoder_targets: torch.Tensor,
                          source_mask: torch.Tensor = None,
                          target_mask: torch.Tensor = None): 
        # run forward and return loss 
        encoded = source_memory.unsqueeze(1)
        source_mask = torch.ones_like(encoded)[:,:,0]
        decoded = self.decoder(inputs=decoder_inputs,
                               source_memory_bank=encoded,
                               source_mask=source_mask,
                               target_mask=target_mask)
        decoded = self.activation(self.dropout(decoded['outputs']))
        output = self.output_layer(decoded)
        output_for_loss = rearrange(output, "b n v -> (b n) v")
        targets = rearrange(decoder_targets, "b n -> (b n)") 
        top_k = torch.argmax(output_for_loss, dim=1)
        print()
        print(f"pred: {top_k[0:10]}")
        print(f"true: {targets[0:10]}")
        print()

        loss = self.loss(output_for_loss, targets) 

        return {"encoder_output": encoded, "output": output, "top_k": top_k, "loss": loss}

    def _test_forward(self,
                      decoder_inputs: torch.Tensor,
                      source_memory: torch.Tensor,
                      source_mask: torch.Tensor,
                      target_mask: torch.Tensor = None):
        # run beam search 
        pass 
