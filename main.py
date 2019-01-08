from features.authors import ExtractAuthorsList
from features.title import CleanTitle, TitleOverlapping
from features.date import PublicationDateDiff
from features.collaboration import CollaborationGraphFeatures, CollaborationFeatures
from features.papers_graph import ExtractGraphPapersFeatures
from features.journal import ExtractJournalFeatures
from features.nlp import ExtractCosineSimilarity

"""Contains pipeline with all the features extractor"""


# load features
pipeline = [
    CleanTitle(),
    TitleOverlapping(),
    PublicationDateDiff(),
    ExtractAuthorsList(),
    CollaborationGraphFeatures(),
    CollaborationFeatures(),
    ExtractGraphPapersFeatures(),
    ExtractJournalFeatures(),
    ExtractCosineSimilarity()
]
