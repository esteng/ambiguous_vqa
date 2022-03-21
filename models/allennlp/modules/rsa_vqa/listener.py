from copy import deepcopy
from typing import Any

import torch
import pdb 

from allennlp.common.registrable import Registrable
from allennlp.modules.seq2seq_encoders.prenorm_transformer_encoder import MisoTransformerEncoder

class BaseListenerModule(torch.nn.Module, Registrable):
    def __init__(self, 
                 encoder: Any,
                 pooling_strategy: str = "mean",
                 dropout: float = 0.1): 
        super(BaseListenerModule, self).__init__()
        self.encoder = encoder 
        self.dropout = torch.nn.Dropout(dropout)
        self.activation = torch.nn.ReLU()
        self.pooling_strategy = pooling_strategy

@BaseListenerModule.register("prenorm_listener")
class PrenormListenerModule(BaseListenerModule):
    def __init__(self, 
                 encoder: MisoTransformerEncoder,
                 pooling_strategy: str = "mean",
                 dropout: float = 0.1): 
        super(PrenormListenerModule, self).__init__(encoder=encoder,
                                                    pooling_strategy=pooling_strategy,
                                                   dropout=dropout)

    def forward(self, 
                fused_representation: torch.Tensor,
                fused_mask: torch.Tensor): 

        encoded = self.encoder(inputs = fused_representation, mask=fused_mask)

        if self.pooling_strategy == "mean": 
            if encoded.shape[1] == 1:
                output = encoded.squeeze(1)
            else:
                output = torch.mean(encoded, dim=1)
            

        return {"output": output}

@BaseListenerModule.register("simple_listener")
class PrenormListenerModule(BaseListenerModule):
    def __init__(self, 
                 encoder_in_dim: int, 
                 encoder_num_layers: int, 
                 encoder_hidden_dim: int,
                 encoder_dropout: float,
                 encoder_activation: str, 
                 pooling_strategy: str = "mean",
                 dropout: float = 0.1): 
        super(PrenormListenerModule, self).__init__(encoder=None,
                                                    pooling_strategy=pooling_strategy,
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


    def forward(self, 
                fused_representation: torch.Tensor,
                fused_mask: torch.Tensor): 

        encoded = self.encoder(fused_representation) 

        if self.pooling_strategy == "mean" and len(encoded.shape) > 2: 
            if encoded.shape[1] == 1:
                output = encoded.squeeze(1)
            else:
                output = torch.mean(encoded, dim=1)
        else:
            output = encoded 

        return {"output": output}