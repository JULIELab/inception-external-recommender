from pathlib import Path
from typing import List, Optional
from cassis import Cas
from transformers import pipeline

from ariadne.classifier import Classifier

from ariadne.contrib.inception_util import create_prediction, SENTENCE_TYPE


class MedicationAnnotator(Classifier):
    def __init__(self, model_directory: Path = None):
        super().__init__(model_directory=model_directory)
        self.nerPipeline = pipeline('ner', model=model_directory, tokenizer=model_directory, aggregation_strategy="simple")

    def predict(self, cas: Cas, layer: str, feature: str, project_id: str, document_id: str, user_id: str):
        for sentence in cas.select(SENTENCE_TYPE):
            text = sentence.get_covered_text()
            nerResult = self.nerPipeline(text)

            for ne in nerResult:
                begin = ne["start"]
                end   = ne["end"]
                label = ne["entity_group"]
                neAnnotation = create_prediction(cas, layer, feature, begin, end, label)
                cas.add(neAnnotation)
