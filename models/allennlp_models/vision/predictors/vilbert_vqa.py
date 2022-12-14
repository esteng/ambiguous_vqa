from typing import List, Dict


import numpy

from allennlp.common.file_utils import cached_path
from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor


@Predictor.register("vilbert_vqa")
class VilbertVqaPredictor(Predictor):
    def predict(self, image: str, question: str) -> JsonDict:
        image = cached_path(image)
        return self.predict_json({"question": question, "image": image})

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        from allennlp_models.vision.dataset_readers.vqav2 import VQAv2Reader
        from allennlp_models.vision import GQAReader

        question = json_dict["question"]
        image = cached_path(json_dict["image"])
        if isinstance(self._dataset_reader, VQAv2Reader) or isinstance(
            self._dataset_reader, GQAReader
        ):
            return self._dataset_reader.text_to_instance(question, image, use_cache=False)
        else:
            raise ValueError(
                f"Dataset reader is of type f{self._dataset_reader.__class__.__name__}. "
                f"Expected {VQAv2Reader.__name__}."
            )

    def predictions_to_labeled_instances(
        self, instance: Instance, outputs: Dict[str, numpy.ndarray]
    ) -> List[Instance]:
        return [instance]  # TODO
