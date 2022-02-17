import pdb 
import torch


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
                fused_mask: torch.Tensor,
                gold_utterance: torch.Tensor=None): 
        if gold_utterance is not None: 
            return self._training_forward(decoder_inputs=gold_utterance,
                                          source_memory=fused_representation, 
                                          source_mask=fused_mask)
        else:
            return self._test_forward(fused_representation)

    def _training_forward(self, 
                          decoder_inputs: torch.Tensor,
                          source_memory: torch.Tensor,
                          source_mask: torch.Tensor,
                          target_mask: torch.Tensor = None): 
        # run forward and return loss 
        encoded = self.encoder(source_memory, source_mask)
        decoded = self.decoder(inputs=decoder_inputs,
                               source_memory_bank=encoded,
                               source_mask=source_mask,
                               target_mask=target_mask)
        decoded = self.activation(self.dropout(decoded))
        output = self.output_layer(decoded)

        return {"output": output}

    def _test_forward(self,
                      decoder_inputs: torch.Tensor,
                      source_memory: torch.Tensor,
                      source_mask: torch.Tensor,
                      target_mask: torch.Tensor = None):
        # run beam search 
        pass 
