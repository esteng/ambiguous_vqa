from typing import Tuple, Dict, Optional
from overrides import overrides
import logging
import copy 
import pdb 

import torch

from allennlp.common.registrable import Registrable
from allennlp.modules import InputVariationalDropout
from allennlp.nn.util import add_positional_features

from allennlp.modules.transformer.miso_transformer_layers import MisoTransformerDecoderLayer, MisoPreNormTransformerDecoderLayer

logger = logging.getLogger(__name__) 

def _get_clones(module, N):
    return torch.nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class MisoTransformerDecoder(torch.nn.Module, Registrable):
    def __init__(self, 
                    input_size: int,
                    hidden_size: int,
                    decoder_layer: MisoTransformerDecoderLayer, 
                    num_layers: int,
                    dropout=0.1):
        super(MisoTransformerDecoder, self).__init__()

        self.input_proj_layer = torch.nn.Linear(input_size, hidden_size)

        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.dropout = InputVariationalDropout(dropout)

        self.prenorm = isinstance(decoder_layer, MisoPreNormTransformerDecoderLayer)

        if self.prenorm:
            self.final_norm = copy.deepcopy(decoder_layer.norm4)

    @overrides
    def forward(self,
                inputs: torch.Tensor,
                source_memory_bank: torch.Tensor,
                source_mask: torch.Tensor,
                target_mask: torch.Tensor) -> Dict: 

        source_padding_mask = None
        target_padding_mask  = None
        if source_mask is not None:
            source_padding_mask = ~source_mask.bool()
        if target_mask is not None:
            target_padding_mask = ~target_mask.bool() 

        # project to correct dimensionality 
        outputs = self.input_proj_layer(inputs)
        # add pos encoding feats 
        outputs = add_positional_features(outputs) 

        # swap to pytorch's batch-second convention 
        outputs = outputs.permute(1, 0, 2)
        source_memory_bank = source_memory_bank.permute(1, 0, 2)

        # get a mask 
        ar_mask = self.make_autoregressive_mask(outputs.shape[0]).to(source_memory_bank.device)

        for i in range(len(self.layers)):
            outputs , __, __ = self.layers[i](outputs, 
                                    source_memory_bank, 
                                    tgt_mask=ar_mask,
                                    #memory_mask=None,
                                    tgt_key_padding_mask=target_padding_mask,
                                    memory_key_padding_mask=source_padding_mask
                                    )

        # do final norm here
        if self.prenorm:
            outputs = self.final_norm(outputs) 

        # switch back from pytorch's absolutely moronic batch-second convention
        outputs = outputs.permute(1, 0, 2)
        source_memory_bank = source_memory_bank.permute(1, 0, 2) 

        return dict(
                outputs=outputs,
                output=outputs[:,-1,:].unsqueeze(1),
                ) 

    def one_step_forward(self,
                         inputs: torch.Tensor,
                         source_memory_bank: torch.Tensor,
                         source_mask: torch.Tensor,
                         total_decoding_steps: int = 0) -> Dict:
        """
        Run a single step decoding.
        :param input_tensor: [batch_size, seq_len, input_vector_dim].
        :param source_memory_bank: [batch_size, source_seq_length, source_vector_dim].
        :param source_mask: [batch_size, source_seq_length].
        :return:
        """
        target_mask = None  
        to_ret = self(inputs, source_memory_bank, source_mask, target_mask)

        return to_ret 

    def make_autoregressive_mask(self,
                                 size: int):
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

@MisoTransformerDecoder.register("transformer_decoder") 
class MisoBaseTransformerDecoder(MisoTransformerDecoder):
    def __init__(self, 
                    input_size: int,
                    hidden_size: int,
                    decoder_layer: MisoTransformerDecoderLayer, 
                    num_layers: int,
                    dropout=0.1):
        super(MisoBaseTransformerDecoder, self).__init__(input_size,
                                                         hidden_size,
                                                        decoder_layer,
                                                        num_layers,
                                                        dropout)
