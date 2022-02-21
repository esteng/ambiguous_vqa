import torch
import pdb 

from allennlp.common.registrable import Registrable
from allennlp.modules.seq2seq_encoders.prenorm_transformer_encoder import MisoTransformerEncoder

class BaseListenerModule(torch.nn.Module, Registrable):
    def __init__(self, 
                 encoder: MisoTransformerEncoder,
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