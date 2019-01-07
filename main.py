from features.authors import ExtractAuthorsList
from features.title import CleanTitle, TitleOverlapping
from features.date import PublicationDateDiff
from features.collaboration import CollaborationDistance, CollaborationFeatures
from features.papers_graph import ExtractGraphPapersFeatures
from features.journal import ExtractJournalFeatures

"""Contains pipeline with all the features extractor"""


# load features
pipeline = [
    CleanTitle(),
    TitleOverlapping(),
    PublicationDateDiff(),
    ExtractAuthorsList(),
    CollaborationDistance(),
    CollaborationFeatures(),
    ExtractGraphPapersFeatures(),
    ExtractJournalFeatures()
]
