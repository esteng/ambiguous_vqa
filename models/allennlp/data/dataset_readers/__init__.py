"""
A :class:`~allennlp.data.dataset_readers.dataset_reader.DatasetReader`
reads a file and converts it to a collection of
:class:`~allennlp.data.instance.Instance` s.
The various subclasses know how to read specific filetypes
and produce datasets in the formats required by specific models.
"""


from allennlp.data.dataset_readers.dataset_reader import (
    DatasetReader,
    WorkerInfo,
)
from allennlp.data.dataset_readers.babi import BabiReader
from allennlp.data.dataset_readers.conll2003 import Conll2003DatasetReader
from allennlp.data.dataset_readers.interleaving_dataset_reader import InterleavingDatasetReader
from allennlp.data.dataset_readers.multitask import MultiTaskDatasetReader
from allennlp.data.dataset_readers.sequence_tagging import SequenceTaggingDatasetReader
from allennlp.data.dataset_readers.sharded_dataset_reader import ShardedDatasetReader
from allennlp.data.dataset_readers.text_classification_json import TextClassificationJsonReader

# try:
from allennlp.data.dataset_readers.vision_reader import VisionReader
from allennlp.data.dataset_readers.vqav2 import VQAv2Reader
print(VQAv2Reader)
from allennlp.data.dataset_readers.visual_entailment import VisualEntailmentReader
# except ModuleNotFoundError as err:
    # if err.name not in ("detectron2", "torchvision"):
        # raise