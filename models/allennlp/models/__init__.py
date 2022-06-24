"""
These submodules contain the classes for AllenNLP models,
all of which are subclasses of `Model`.
"""

from allennlp.models.model import Model
from allennlp.models.archival import archive_model, load_archive, Archive
from allennlp.models.basic_classifier import BasicClassifier
from allennlp.models.multitask import MultiTaskModel
from allennlp.models.simple_tagger import SimpleTagger
from allennlp.models.vilbert_vqa import VqaVilbert
from allennlp.models.simple_model import SimpleDebugModel
from allennlp.models.rsa_vqa import RSAVQAModel
from allennlp.models.debug_rsa import DebugRSAVQAModel
from allennlp.models.img_ans_to_question import ImageAnswer2QuestionModel
from allennlp.models.t5_img_ans2question import T5ImageAnswer2QuestionModel
