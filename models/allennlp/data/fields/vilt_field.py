from typing import Dict, Any, Union, Optional, List

import torch
import numpy as np
from overrides import overrides
from transformers import  ViltProcessor
from PIL import Image

from allennlp.data.fields.field import DataArray
from allennlp.data.fields.metadata_field import MetadataField

class ViltField(MetadataField):
    """
    A class representing a tensor, which could have arbitrary dimensions.
    A batch of these tensors are padded to the max dimension length in the batch
    for each dimension.
    """

    __slots__ = ["metadata", "vilt_processor", "vilt_half_precision"]
    def __init__(self, metadata: Any, 
                vilt_processor: ViltProcessor,
                vilt_half_precision: bool = True) -> None:
        super(ViltField, self).__init__(metadata)
        self.metadata = metadata
        self.vilt_processor = vilt_processor 
        self.vilt_half_precision = vilt_half_precision

    @overrides
    def batch_tensors(self, tensor_list: List[DataArray]) -> List[DataArray]:  # type: ignore
        
        texts = []
        images = [] 
        for tensor in tensor_list:
            text = tensor['text']
            texts.append(text)
            image = tensor['image']
            image_data = Image.open(image).convert("RGB")
            images.append(image_data)


        processed = self.vilt_processor(text = texts, 
                                        images=images, 
                                        return_tensors='pt', 
                                        padding=True)

        to_ret = {}
        for k, v in processed.items():
            if self.vilt_half_precision and (isinstance(v, torch.FloatTensor) or isinstance(v, torch.cuda.FloatTensor)):
                processed[k] = v.half()
            to_ret[k]  = processed[k]
        return to_ret 