from cassis import Cas, TypeSystem
from ariadne.contrib.inception_util import SENTENCE_TYPE, IS_PREDICTION
from ariadne.contrib.julielab import MEDICATION_SPECIFICATION_TYPES, DRUG_TYPE, CONCEPT_TYPE
from ariadne.contrib.julielab import MedicationAnnotator

FEATURE_SOURCE = "source"

def test_medication_annotator():
	annotator = MedicationAnnotator("I2B2-2009")
	typesystem = build_typesystem()
	sentenceType = typesystem.get_type(SENTENCE_TYPE)
	extCoreType = typesystem.get_type(CONCEPT_TYPE)

	cas = Cas(typesystem=typesystem)
	cas.sofa_string = "The patient was given 2 units paracetamole twice a day for his ongoing headache. He also received ibuprophene once a day for his sprained ankle."
	cas.add(sentenceType(begin=0,end=80))
	cas.add(sentenceType(begin=81,end=len(cas.sofa_string)))
	annotator.predict(cas, CONCEPT_TYPE, FEATURE_SOURCE, "TestProject", "TestDocId", "TestUser")
	numNe = 0
	for annotation in sorted(cas.select(CONCEPT_TYPE), key=lambda a: a.begin):
	    numNe = numNe + 1
	    if numNe == 1:
	       assert annotation.get_covered_text() == "2 units"
	       assert annotation.begin == 22
	       assert annotation.end == 29
	    if numNe == 7:
	        assert annotation.get_covered_text() ==  "his sprained ankle."
	        assert annotation.begin == 125
	        assert annotation.end == 144
	assert numNe == 7

def build_typesystem() -> TypeSystem:
	typesystem = TypeSystem()
	typesystem.create_type(SENTENCE_TYPE)
	conceptType = typesystem.create_type(CONCEPT_TYPE)
	typesystem.create_feature(conceptType, FEATURE_SOURCE, "uima.cas.String")
	for annotationTypeName in MEDICATION_SPECIFICATION_TYPES + [DRUG_TYPE]:
		annotationType = typesystem.create_type(annotationTypeName, supertypeName=CONCEPT_TYPE)	
		typesystem.create_feature(annotationType, FEATURE_SOURCE, "uima.cas.String")
	return typesystem