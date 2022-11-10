from ariadne.contrib.julielab import MedicationRelationAnnotator
from ariadne.contrib.julielab import MEDICATION_SPECIFICATION_TYPES, DRUG_TYPE, EXT_CORE_TYPE
from cassis import Cas, TypeSystem
from ariadne.contrib.inception_util import SENTENCE_TYPE

FREQ_TYPE = "de.averbis.types.health.DoseFrequency"
DOSE_TYPE = "de.averbis.types.health.DoseForm"

FEATURE_SOURCE = "source"

RELATION_TYPE = "de.averbis.types.health.Relation"
RELATION_FEATURE_SUBJECT = "subject"
RELATION_FEATURE_OBJECT = "object"


def test_relation_annotator():
	annotator = MedicationRelationAnnotator()
	typesystem = build_typesystem()
	SentenceType = typesystem.get_type(SENTENCE_TYPE)
	DrugType = typesystem.get_type(DRUG_TYPE)
	FreqType = typesystem.get_type(FREQ_TYPE)
	DoseType = typesystem.get_type(DOSE_TYPE)

	cas = Cas(typesystem=typesystem)
	cas.sofa_string = ("The patient was given 2 units paracetamole twice a day for his ongoing headache."
	" He also received ibuprophene once a day, 1 unit and 3 units once a day of aspirin for his sprained ankle.")
	cas.add(SentenceType(begin=0,end=80))
	cas.add(SentenceType(begin=81,end=len(cas.sofa_string)))
	twoUnits = DoseType(begin=22,end=29)
	paracetamole = DrugType(begin=30,end=42)
	twiceADay = FreqType(begin=43,end=54)
	cas.add(twoUnits)
	cas.add(paracetamole)
	cas.add(twiceADay)

	ibuprophene = DrugType(begin=98,end=109)
	onceADay = FreqType(begin=110,end=120)
	oneUnit = DoseType(begin=122,end=128)
	threeUnits = DoseType(begin=133,end=140)
	onceADay2 = FreqType(begin=141,end=151)
	aspirin = DrugType(begin=155,end=162)
	cas.add(ibuprophene)
	cas.add(onceADay)
	cas.add(oneUnit)
	cas.add(threeUnits)
	cas.add(onceADay2)
	cas.add(aspirin)

	annotator.predict(cas, None, None, None, None, None)

	relations = cas.select(RELATION_TYPE)
	assert len(relations) == 6
	assert relations[0].subject == paracetamole
	assert relations[0].object == twoUnits
	assert relations[1].subject == paracetamole
	assert relations[1].object == twiceADay
	assert relations[2].subject == ibuprophene
	assert relations[2].object == onceADay
	assert relations[3].subject == ibuprophene
	assert relations[3].object == oneUnit
	assert relations[4].subject == aspirin
	assert relations[4].object == threeUnits
	assert relations[5].subject == aspirin
	assert relations[5].object == onceADay2



def build_typesystem() -> TypeSystem:
	typesystem = TypeSystem()
	typesystem.create_type(SENTENCE_TYPE)
	typesystem.create_type(EXT_CORE_TYPE)
	for annotationTypeName in MEDICATION_SPECIFICATION_TYPES + [DRUG_TYPE]:
		annotationType = typesystem.create_type(annotationTypeName, supertypeName=EXT_CORE_TYPE)
		typesystem.create_feature(annotationType, FEATURE_SOURCE, "uima.cas.String")
	relationType = typesystem.create_type(RELATION_TYPE)
	typesystem.create_feature(relationType, RELATION_FEATURE_SUBJECT, EXT_CORE_TYPE)
	typesystem.create_feature(relationType, RELATION_FEATURE_OBJECT, EXT_CORE_TYPE)
	
	return typesystem

