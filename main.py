from core.io.loader import get_data_with_node_information
from features.authors import ExtractAuthorsList
from features.title import CleanTitle, TitleOverlapping
from features.date import PublicationDateDiff
from features.collaboration import CollaborationDistance, CollaborationFeatures
from features.papers_graph import ExtractGraphPapersFeatures
from features.journal import ExtractJournalFeatures

"""
This code aims to extract features from train data
"""

# load data
train = get_data_with_node_information('train')

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

for task in pipeline:
    task.fit(train)
    train = task.transform(train)
