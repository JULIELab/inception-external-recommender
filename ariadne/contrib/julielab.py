from pathlib import Path
from typing import List, Optional
from cassis import Cas
from transformers import pipeline

import logging


from ariadne.classifier import Classifier

from ariadne.contrib.inception_util import create_prediction, SENTENCE_TYPE

logger = logging.getLogger(__name__)


class MedicationAnnotator(Classifier):
    def __init__(self, model_directory: Path = None):
        super().__init__(model_directory=model_directory)
        self.nerPipeline = pipeline('ner', model=model_directory, tokenizer=model_directory, aggregation_strategy="first")

    def predict(self, cas: Cas, layer: str, feature: str, project_id: str, document_id: str, user_id: str):
        model = self._load_model(user_id)

        if model is None:
            logger.debug("No trained model ready yet!")
            # return

        for sentence in cas.select(SENTENCE_TYPE):
            text = sentence.get_covered_text()
            logger.debug(layer)
			#logger.debug(feature)
            nerResult = self.nerPipeline(text)

            for ne in nerResult:
                logger.debug(f"Output [%s]", str(ne))
                begin = ne["start"] + sentence.begin
                end   = ne["end"] + sentence.begin
                label = ne["entity_group"]

                if label == 'Strength':
	                neAnnotation = create_prediction(cas, 'de.averbis.types.health.Strength', 'source', begin, end, label)
                elif label == 'Drug' or label == 'medication':
                    neAnnotation = create_prediction(cas, 'de.averbis.types.health.Ingredient', 'source', begin, end, label)
                elif label == 'Frequency' or label == 'frequency':
                    neAnnotation = create_prediction(cas, 'de.averbis.types.health.DoseFrequency', 'source', begin, end, label)
                elif label == 'Form' or label == 'dosage':
                    neAnnotation = create_prediction(cas, 'de.averbis.types.health.DoseForm', 'source', begin, end, label)
                elif label == 'ADE':
                    neAnnotation = create_prediction(cas, 'de.averbis.types.health.Diagnosis', 'source', begin, end, label)
                elif label == 'Duration' or label == 'duration':
                    neAnnotation = create_prediction(cas, 'de.averbis.types.health.DateInterval', '', begin, end, label)
                elif label == 'Dosage':
#                    neAnnotation = create_prediction(cas, 'de.averbis.textanalysis.types.health.Dosage', '', begin, end, label)', 'componentId', begin, end, label)
                    neAnnotation = create_prediction(cas, 'de.averbis.types.health.Measurement', '', begin, end, label)
#                elif label == 'Reason':
#                    neAnnotation = create_prediction(cas, 'de.averbis.types.health.Concept', 'source', begin, end, label)
                elif label == 'Route' or label == 'mode':
                    neAnnotation = create_prediction(cas, 'de.averbis.types.health.Procedure', 'source', begin, end, label)
                else:
	                neAnnotation = create_prediction(cas, layer, feature, begin, end, label)

#                neAnnotation = create_prediction(cas, layer, feature, begin, end, label)
                cas.add_annotation(neAnnotation)
