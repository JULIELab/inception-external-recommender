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

DRUG_TYPE = 'de.averbis.types.health.Ingredient'
MEDICATION_SPECIFICATION_TYPES = [
            'de.averbis.types.health.Strength',
            'de.averbis.types.health.DoseFrequency',
            'de.averbis.types.health.DoseForm',
            'de.averbis.types.health.Diagnosis',
            'de.averbis.types.health.DateInterval',
            'de.averbis.types.health.Measurement',
            'de.averbis.types.health.Procedure'
        ]
RELATION_TYPE = "de.averbis.types.health.Relation"
RELATION_FEATURE_SUBJECT = "subject"
RELATION_FEATURE_OBJECT = "object"

EXT_CORE_TYPE = "de.averbis.types.health.ExternalCoreAnnotation"
CONCEPT_TYPE = "de.averbis.types.health.Concept"

class MedicationRelationAnnotator(Classifier):
    """
    A rule-based relation annotator for medication expressions.
    A medication, in the sense of 'prescribing someone a drug', is expressed
    in multiple facets like which drug, how often to take, how much to take,
    why to take etc. All of these aspects are captured in the MedicationAnnotator
    that should be applied to the CAS before this annotator.

    This annotator creates binary relations of type RELATION_TYPE between the drug
    mention and other medication aspects, if found. In the code, we call those aspects
    'specifications' (how much, how often, ...).
    The relations are created sentence-wise and always between entities within the same
    sentence. The drug is always the subject of the relation, the other aspect is the object.
    We currently omit the other features (predicate, value).
    The algorithm creates relations between a drug and its previously mentioned other medication aspects,
    if a specific aspect has not already been used for a relation of a previous drug.
    It also created relations for medication aspects succeeding a drug until another drug occurs.

    Medication aspects between two drug mentions are assigned to the first mention as long as it
    does not already have a relation to another aspect of the same type.

    Thus, each drug is only assigned a single dosage form, dosage frequency etc.
    """
    def dotToUnderline(self, typeName):
        return typeName.replace(".","_")

    def createMedicationRelation(self, drug, specification, cas):
        RelationType = cas.typesystem.get_type(RELATION_TYPE)
        cas.add(RelationType(begin=drug.begin,end=drug.end,subject=drug,object=specification))


    def predict(self, cas: Cas, layer: str, feature: str, project_id: str, document_id: str, user_id: str):
        for sentence in cas.select(SENTENCE_TYPE):
            # This set stores the medication specification annotations that have not been used in a
            # relation yet while we traverse all medication entities in order of appearance in the sentence.
            availableSpecifications = {}
            # We create a list of all medication annotations. This happens in the next few lines.
            annotations = []
            for medicationSpecificationTypeName in MEDICATION_SPECIFICATION_TYPES:
                for medicationSpecificationAnnotation in cas.select_covered(medicationSpecificationTypeName, sentence):
                    annotations.append(medicationSpecificationAnnotation)
            for drugAnnotation in cas.select_covered(DRUG_TYPE, sentence):
                annotations.append(drugAnnotation)
            # Sort all the medication-related annotations by begin offset
            annotations.sort(key=lambda a: a.begin)
           
            # We now have all the medication annotations. We can now traverse them, adding
            # to the available specifications and assigning them to drugs as they occur in the
            # sequence.
            lastDrug = None
            # Store the specification types we have already assigned to the last seen drug.
            # This set is reset each time a drug occurs, prohibiting long-range relations
            # over other drugs.
            specsAssignedToLastDrug = set()
            for a in annotations:
                # There are two basic cases in this loop: The current annotation is a drug or it is not.
                # If it not, we assign the medication specification to a previously mentioned drug, if
                # available and if that drug does not already have a specification of the current type.
                # Otherwise, we save the medication specification annotation for a drug annotation to come.
                annotationClassName = type(a).__name__
                if not annotationClassName == self.dotToUnderline(DRUG_TYPE):
                    if lastDrug and not annotationClassName in specsAssignedToLastDrug:
                        # The previous drug does not have a specification of the current type.
                        # We create a relation from the previous drug annotation.
                        self.createMedicationRelation(lastDrug, a, cas)
                        specsAssignedToLastDrug.add(annotationClassName)
                    else:
                        # Save the annotation for the next drug.
                        # We use the last occurrence of each specification type. If there are
                        # multiple entities of the same medication aspect type but we didn't
                        # come across a drug yet, we discard the previous annotation.
                        availableSpecifications[annotationClassName] = a
                else:
                    # This is a drug. Create relations to previous, available medication
                    # specification entities.
                    specsAssignedToLastDrug = set()
                    for (specName, specsOfName) in availableSpecifications.items():
                        self.createMedicationRelation(a, specsOfName, cas)
                        specsAssignedToLastDrug.add(specName)
                    lastDrug = a