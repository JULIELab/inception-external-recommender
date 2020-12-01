from ariadne.contrib import *
from ariadne.contrib.adapters import AdapterSentenceClassifier
from ariadne.server import Server
from ariadne.util import setup_logging

setup_logging()

server = Server()
# server.add_classifier("spacy_ner", SpacyNerClassifier("en"))
# server.add_classifier("spacy_pos", SpacyPosClassifier("en"))
# server.add_classifier("sklearn_sentence", SklearnSentenceClassifier())
# server.add_classifier("jieba", JiebaSegmenter())
# server.add_classifier("stemmer", NltkStemmer())
# server.add_classifier("leven", LevenshteinStringMatcher())
# server.add_classifier("sbert", SbertSentenceClassifier())
# server.add_classifier(
#     "adapter_pos",
#     AdapterSequenceTagger(
#         base_model_name="bert-base-uncased",
#         adapter_name="pos/ldc2012t13@vblagoje",
#         labels=[
#             "ADJ",
#             "ADP",
#             "ADV",
#             "AUX",
#             "CCONJ",
#             "DET",
#             "INTJ",
#             "NOUN",
#             "NUM",
#             "PART",
#             "PRON",
#             "PROPN",
#             "PUNCT",
#             "SCONJ",
#             "SYM",
#             "VERB",
#             "X",
#         ],
#     ),
# )
#
# server.add_classifier(
#     "adapter_sent",
#     AdapterSentenceClassifier(
#         base_model_name="bert-base-uncased", adapter_name="sentiment/sst-2@ukp", labels=["negative", "positive"]
#     ),
# )

server.start(debug=True, port=40022)
