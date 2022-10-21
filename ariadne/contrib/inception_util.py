from cassis import Cas
from cassis.typesystem import FeatureStructure

SENTENCE_TYPE = "de.averbis.extraction.types.Sentence"
TOKEN_TYPE = "de.averbis.extraction.types.Token"
IS_PREDICTION = "inception_internal_predicted"


def create_prediction(cas: Cas, layer: str, feature: str, begin: int, end: int, label: str) -> FeatureStructure:
    AnnotationType = cas.typesystem.get_type(layer)

    # fields = {"begin": begin, "end": end, IS_PREDICTION: True, feature: label}
    if len(feature) > 1:
        fields = {"begin": begin, "end": end, feature: label}
    else:
        fields = {"begin": begin, "end": end}
    prediction = AnnotationType(**fields)
    return prediction
