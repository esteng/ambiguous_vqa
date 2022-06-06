from copy import deepcopy 
from typing import List, Any
import pdb 
import pathlib
import time 

import torch
from transformers import AutoModel, AutoTokenizer, CLIPModel, CLIPProcessor, ViltProcessor, ViltForQuestionAnswering, ViltModel, ViltFeatureExtractor, BertTokenizerFast
from PIL import Image

from allennlp.common import Registrable, Params
from allennlp.data.fields.text_field import TextFieldTensors
from allennlp.modules.transformer import (
    TextEmbeddings,
    ImageFeatureEmbeddings,
    BiModalEncoder,
    TransformerPooler,
)


class VisionLanguageEncoder(torch.nn.Module, Registrable):
    def __init__(self):
        super(VisionLanguageEncoder, self).__init__()

@VisionLanguageEncoder.register("vilbert")
@VisionLanguageEncoder.register("vilbert_from_huggingface", constructor="from_huggingface_model_name")
class ViLBERTLanguageEncoder(VisionLanguageEncoder):
    def __init__(self,
                 embeddings: TextEmbeddings,
                 image_embeddings: ImageFeatureEmbeddings,
                 encoder: BiModalEncoder,
                 t_pooler: TransformerPooler,
                 v_pooler: TransformerPooler,
                 tune_bert: bool = False,
                 tune_images: bool = False,
                 dropout: float = 0.2,
                 fusion_method: str = "mul",
                 ):
        super(ViLBERTLanguageEncoder, self).__init__()
        self.embeddings = embeddings
        self.image_embeddings = image_embeddings
        self.encoder = encoder 
        self.t_pooler = t_pooler
        self.v_pooler = v_pooler 
        self.tune_bert = tune_bert
        self.tune_images = tune_images
        self.dropout = torch.nn.Dropout(dropout) 
        self.fusion_method = fusion_method 

    def forward(self, 
                box_features: torch.Tensor,
                box_coordinates: torch.Tensor,
                question: TextFieldTensors,
                question_input: torch.Tensor = None,
                ):

        batch_size, _, feature_size = box_features.size()
        # TODO(mattg): have this make fewer assumptions.
        input_ids = question["tokens"]["token_ids"]
        token_type_ids = question["tokens"]["type_ids"]
        attention_mask = question["tokens"]["mask"]
        
        def get_embeddings(input_ids, token_type_ids, question_input):
            question_embedded_input = self.embeddings(input_ids, token_type_ids)
            num_tokens = question_embedded_input.size(1)
            # pdb.set_trace() 
            # if question_input is not None:
            #     # question_embedded_for_teacher_forcing = self.token_embeddings(question_input['tokens']['tokens'], question_input['tokens']['type_ids'])
            # else:
            #     question_embedded_for_teacher_forcing = None
            return question_embedded_input, num_tokens # , question_embedded_for_teacher_forcing

        # Get text embedding 
        if self.tune_bert:
            question_embedded_input, num_tokens = get_embeddings(input_ids, token_type_ids, question_input)
        else:
            with torch.no_grad():
                question_embedded_input, num_tokens = get_embeddings(input_ids, token_type_ids, question_input)

        # get image region embedding 
        if self.tune_images: 
            v_embedding_output = self.image_embeddings(box_features, box_coordinates)
        else:
            with torch.no_grad():
                # (batch_size, num_boxes, image_embedding_dim)
                v_embedding_output = self.image_embeddings(box_features, box_coordinates)

        # All batch instances will always have the same number of images and boxes, so no masking
        # is necessary, and this is just a tensor of ones.
        image_attention_mask = torch.ones_like(box_coordinates[:, :, 0])

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of
        # causal attention used in OpenAI GPT, we just need to prepare the
        # broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2).float().log()
        extended_image_attention_mask = image_attention_mask.unsqueeze(1).unsqueeze(2).float().log()

        extended_co_attention_mask = torch.zeros(
            batch_size,
            feature_size,
            num_tokens,
            dtype=extended_image_attention_mask.dtype,
        )


        encoded_layers_t, encoded_layers_v = self.encoder(
            question_embedded_input,
            v_embedding_output,
            extended_attention_mask,
            extended_image_attention_mask,
            extended_co_attention_mask,
        )

        # Pooling and fusing into one single representation 
        
        sequence_output_t = encoded_layers_t[:, :, :, -1]
        sequence_output_v = encoded_layers_v[:, :, :, -1]

        pooled_output_t = self.t_pooler(sequence_output_t)
        pooled_output_v = self.v_pooler(sequence_output_v)

        if self.fusion_method == "sum":
            pooled_output = self.dropout(pooled_output_t + pooled_output_v)
        elif self.fusion_method == "mul":
            pooled_output = self.dropout(pooled_output_t * pooled_output_v)
        else:
            raise ValueError(f"Fusion method '{self.fusion_method}' not supported")

        return pooled_output, sequence_output_t

    @classmethod
    def from_huggingface_model_name(
            cls,
            text_model_name: str,
            image_feature_dim: int,
            image_hidden_size: int,
            image_num_hidden_layers: int,
            image_num_attention_heads: int, 
            combined_hidden_size: int,
            combined_num_attention_heads: int, 
            image_intermediate_size: int,
            image_attention_dropout: float,
            text_biattention_id: List[int],
            image_biattention_id: List[int],
            text_fixed_layer: int,
            image_fixed_layer: int,
            pooled_output_dim: int,
            tune_bert: bool = False,
            tune_images: bool = False,
            image_hidden_dropout: float = 0.2,
            dropout: float = 0.2,
            fusion_method: str = "mul",
            ):

        transformer = AutoModel.from_pretrained(text_model_name)
        embeddings = deepcopy(transformer.embeddings)

        image_embeddings = ImageFeatureEmbeddings(
            feature_dim=image_feature_dim,
            hidden_dim=image_hidden_size,
            dropout=image_hidden_dropout,
        )

        encoder = BiModalEncoder.from_pretrained_module(
            pretrained_module=transformer,
            num_hidden_layers2=image_num_hidden_layers,
            hidden_size2=image_hidden_size,
            num_attention_heads2=image_num_attention_heads,
            combined_hidden_size=combined_hidden_size,
            combined_num_attention_heads=combined_num_attention_heads,
            intermediate_size2=image_intermediate_size,
            attention_dropout2=image_attention_dropout,
            hidden_dropout2=image_hidden_dropout,
            biattention_id1=text_biattention_id,
            biattention_id2=image_biattention_id,
            fixed_layer1=text_fixed_layer,
            fixed_layer2=image_fixed_layer,
        )


        t_pooler = TransformerPooler(encoder.hidden_size1, pooled_output_dim)
        v_pooler = TransformerPooler(encoder.hidden_size2, pooled_output_dim)

        return cls(embeddings=embeddings,
                   image_embeddings=image_embeddings,
                   encoder=encoder,
                   t_pooler=t_pooler,
                   v_pooler=v_pooler,
                   tune_bert=tune_bert,
                   tune_images=tune_images,
                   dropout=dropout,
                   fusion_method=fusion_method)


@VisionLanguageEncoder.register("clip")
class CLIPLanguageEncoder(VisionLanguageEncoder):
    def __init__(self, model_name: str):
        super(CLIPLanguageEncoder, self).__init__()
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.projection_dim = self.model.config.projection_dim

    def forward(self, text_batch, image_batch): 
        # try:
        images = [Image.open(img_path).convert("RGB") for img_path in image_batch]
        inputs = self.processor(text = text_batch, images = images, return_tensors="pt", padding=True).to(self.model.device)
        # except: 
            # pdb.set_trace() 
        outputs = self.model(**inputs) 
        combined_pooled = torch.cat([outputs['image_embeds'].unsqueeze(1), 
                                   outputs['text_embeds'].unsqueeze(1)], dim=1) 

        # TODO (elias) later on could be attention 
        combined_pooled = torch.mean(combined_pooled, dim=1).squeeze(1)
        return combined_pooled, outputs['text_model_output']['last_hidden_state']

@VisionLanguageEncoder.register("vilt")
class ViLTLanguageEncoder(VisionLanguageEncoder):
    def __init__(self, model_name: str, half_precision: bool = False, mlm_prob: float = None):
        super(ViLTLanguageEncoder, self).__init__()
        self.half_precision = half_precision
        if half_precision: 
            self.model = ViltModel.from_pretrained(model_name).half()
        else:
            self.model = ViltModel.from_pretrained(model_name)
        self.processor = ViltProcessor.from_pretrained(model_name)
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
        self.feature_extractor = ViltFeatureExtractor.from_pretrained(model_name)
        self.projection_dim = self.model.config.hidden_size
        self.mlm_prob = mlm_prob

    def supposed_to_be_fast_forward(self, input_ids, token_type_ids, attention_mask, pixel_values, pixel_mask):
        outputs = self.model(input_ids=input_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask,
                            pixel_values=pixel_values,
                            pixel_mask=pixel_mask)

        combined_pooled = outputs['pooler_output'].float()
        return combined_pooled, outputs['last_hidden_state']

    def mask_random_tokens(self, tokenized):
        """add MASK token to tokenized batch"""
        num_tokens = tokenized['attention_mask'].sum(dim=1) - 1
        num_tokens_to_mask = (self.mlm_prob * num_tokens).int()
        # get indices to mask 
        indices_to_mask = [torch.randint(0, tokenized['input_ids'][i].size(0), (num_tokens_to_mask[i],)).tolist() for i in range(len(tokenized['input_ids']))]

        # mask indices 
        for i in range(len(tokenized['input_ids'])):
            tokenized['input_ids'][i][indices_to_mask[i]] = self.tokenizer.mask_token_id
        # pdb.set_trace()
        return tokenized


    def forward(self, text_batch, image_batch):
        images = [Image.open(img_path).convert("RGB") for img_path in image_batch]
        # inputs = self.processor(text = text_batch, images = images, return_tensors="pt", padding=True).to(self.model.device)
        
        encoding = self.tokenizer(
            text=text_batch,
            padding=True,
            return_tensors='pt',
        )
        # pdb.set_trace() 
        if self.mlm_prob is not None:
            encoding = self.mask_random_tokens(encoding)

        # add pixel_values + pixel_mask
        encoding_feature_extractor = self.feature_extractor(images, return_tensors="pt")
        encoding.update(encoding_feature_extractor)
        inputs = encoding.to(self.model.device)

        if self.half_precision:
            for k, v in inputs.items():
                if isinstance(v, torch.FloatTensor) or isinstance(v, torch.cuda.FloatTensor):
                    inputs[k] = v.half()
        outputs = self.model(**inputs) 

        combined_pooled = outputs['pooler_output'].float()
        return combined_pooled, outputs['last_hidden_state']


@VisionLanguageEncoder.register("vilt_classifier")
class ClassifierViLTLanguageEncoder(VisionLanguageEncoder):
    def __init__(self, model_name: str, 
                       half_precision: bool = False):
        super(ClassifierViLTLanguageEncoder, self).__init__()
        self.half_precision = half_precision
        if half_precision: 
            self.model = ViltForQuestionAnswering.from_pretrained(model_name).half()
        else:
            self.model = ViltForQuestionAnswering.from_pretrained(model_name)
        self.processor = ViltProcessor.from_pretrained(model_name)
        self.projection_dim = self.model.config.hidden_size

    # def convert_to_vocab_labels(self, logits):
    #     # batch_size, num_classes 
    #     pred_labels = logits.argmax(dim=-1)
    #     # batch_size
    #     pred_tokens = [self.model.config.id2label[idx] for idx in pred_labels]
    #     pred_vocab_tokens = [self.vocab.get_token_index(x) for x in pred_tokens]
    #     return torch.tensor(pred_vocab_tokens)

    # def forward(self, text_batch, image_batch): 
    def forward(self, input_ids, token_type_ids, attention_mask, pixel_values, pixel_mask):
        # images = [Image.open(img_path) for img_path in image_batch]
        # inputs = self.processor(text = text_batch, images = images, return_tensors="pt", padding=True).to(self.model.device)
        # if self.half_precision:
            # for k, v in inputs.items():
                # if isinstance(v, torch.FloatTensor) or isinstance(v, torch.cuda.FloatTensor):
                    # inputs[k] = v.half()
        # outputs = self.model(**inputs) 
        outputs = self.model(input_ids=input_ids,
                             token_type_ids=token_type_ids,
                             attention_mask=attention_mask,
                             pixel_values=pixel_values,
                             pixel_mask=pixel_mask)
        return outputs['logits']



@VisionLanguageEncoder.register("debug_l_only")
@VisionLanguageEncoder.register("debug_l_only_from_huggingface", constructor="from_huggingface_model_name")
class DebugLanguageEncoder(VisionLanguageEncoder):
    def __init__(self,
                 embeddings: TextEmbeddings,
                 image_embeddings: ImageFeatureEmbeddings,
                 encoder: BiModalEncoder,
                 t_pooler: TransformerPooler,
                 v_pooler: TransformerPooler,
                 tune_bert: bool = False,
                 tune_images: bool = False,
                 dropout: float = 0.2,
                 fusion_method: str = "mul",
                 ):
        super(DebugLanguageEncoder, self).__init__()
        self.embeddings = embeddings
        self.image_embeddings = image_embeddings
        self.encoder = encoder 
        self.t_pooler = t_pooler
        self.v_pooler = v_pooler 
        self.tune_bert = tune_bert
        self.tune_images = tune_images
        self.dropout = torch.nn.Dropout(dropout) 
        self.fusion_method = fusion_method 

    def forward(self, 
                box_features: torch.Tensor,
                box_coordinates: torch.Tensor,
                question: TextFieldTensors,
                question_input: torch.Tensor = None,
                ):

        batch_size, _, feature_size = box_features.size()
        # TODO(mattg): have this make fewer assumptions.
        input_ids = question["tokens"]["token_ids"]
        token_type_ids = question["tokens"]["type_ids"]
        attention_mask = question["tokens"]["mask"]
        
        def get_embeddings(input_ids, token_type_ids, question_input):
            question_embedded_input = self.embeddings(input_ids, token_type_ids)
            num_tokens = question_embedded_input.size(1)
            return question_embedded_input, num_tokens # , question_embedded_for_teacher_forcing

        # Get text embedding 
        with torch.no_grad():
            question_embedded_input, num_tokens = get_embeddings(input_ids, token_type_ids, question_input)

        return torch.mean(question_embedded_input, dim=1), None     

    @classmethod
    def from_huggingface_model_name(
        cls,
        text_model_name: str,
        image_feature_dim: int,
        image_hidden_size: int,
        image_num_hidden_layers: int,
        image_num_attention_heads: int, 
        combined_hidden_size: int,
        combined_num_attention_heads: int, 
        image_intermediate_size: int,
        image_attention_dropout: float,
        text_biattention_id: List[int],
        image_biattention_id: List[int],
        text_fixed_layer: int,
        image_fixed_layer: int,
        pooled_output_dim: int,
        tune_bert: bool = False,
        tune_images: bool = False,
        image_hidden_dropout: float = 0.2,
        dropout: float = 0.2,
        fusion_method: str = "mul",
        ):

        transformer = AutoModel.from_pretrained(text_model_name)
        embeddings = deepcopy(transformer.embeddings)

        image_embeddings = ImageFeatureEmbeddings(
            feature_dim=image_feature_dim,
            hidden_dim=image_hidden_size,
            dropout=image_hidden_dropout,
        )

        encoder = BiModalEncoder.from_pretrained_module(
            pretrained_module=transformer,
            num_hidden_layers2=image_num_hidden_layers,
            hidden_size2=image_hidden_size,
            num_attention_heads2=image_num_attention_heads,
            combined_hidden_size=combined_hidden_size,
            combined_num_attention_heads=combined_num_attention_heads,
            intermediate_size2=image_intermediate_size,
            attention_dropout2=image_attention_dropout,
            hidden_dropout2=image_hidden_dropout,
            biattention_id1=text_biattention_id,
            biattention_id2=image_biattention_id,
            fixed_layer1=text_fixed_layer,
            fixed_layer2=image_fixed_layer,
        )


        t_pooler = TransformerPooler(encoder.hidden_size1, pooled_output_dim)
        v_pooler = TransformerPooler(encoder.hidden_size2, pooled_output_dim)

        return cls(embeddings=embeddings,
                    image_embeddings=image_embeddings,
                    encoder=encoder,
                    t_pooler=t_pooler,
                    v_pooler=v_pooler,
                    tune_bert=tune_bert,
                    tune_images=tune_images,
                    dropout=dropout,
                    fusion_method=fusion_method)