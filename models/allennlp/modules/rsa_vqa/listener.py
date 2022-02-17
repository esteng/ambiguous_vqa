import torch


from allennlp.common.registrable import Registrable
from allennlp.modules.seq2seq_encoders.prenorm_transformer_encoder import MisoTransformerEncoder

class BaseListenerModule(torch.nn.Module, Registrable):
    def __init__(self, 
                 encoder: MisoTransformerEncoder,
                 vocab_size: int,
                 dropout: float = 0.1): 
        super(BaseListenerModule, self).__init__()
        self.encoder = encoder 
        self.dropout = torch.nn.Dropout(dropout)
        self.activation = torch.nn.ReLU()


@BaseListenerModule.register("prenorm_listener")
class PrenormListenerModule(BaseListenerModule):
    def __init__(self, 
                 encoder: MisoTransformerEncoder,
                 vocab_size: int,
                 dropout: float = 0.1): 
        super(PrenormListenerModule, self).__init__(encoder=encoder,
                                                   vocab_size=vocab_size,
                                                   dropout=dropout)

    def forward(self, 
                fused_representation: torch.Tensor,
                fused_mask: torch.Tensor,
                gold_utterance: torch.Tensor=None): 
        if gold_utterance is not None: 
            return self._training_forward(gold_utterance,
                                          fused_representation, 
                                          fused_mask, 
                                          gold_utterance)
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