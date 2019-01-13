from datetime import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from core.io.loader import get_data_with_node_information
from features.authors import ExtractAuthorsList
from features.collaboration import CollaborationGraphFeatures, CollaborationFeatures
from features.date import PublicationDateDiff
from features.features_extractor import FeaturesExtractor
from features.journal import ExtractJournalFeatures
from features.nlp import ExtractCosineSimilarity
from features.papers_graph import ExtractGraphPapersFeatures
from features.processing import features_processing
from features.title import CleanTitle, TitleOverlapping
from models.cross_validation import cross_validation
from plot.features_importances import plot_features_importance

DATA_DIR = './data'

print('loading data')
# Load data
train = get_data_with_node_information('train', data_dir=DATA_DIR)
test = get_data_with_node_information('test', data_dir=DATA_DIR)
print('data loaded')

print('extracting features')
# Extract features
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
features_extractor = FeaturesExtractor(pipeline)
train = features_extractor.fit_transform(train)
test = features_extractor.transform(test)

print('features extracted')


print('selecting and processing features')

# Select columns used to train model and final processing
selected_columns = [
    'title_overlap',
    'publication_date_diff',
    'common_authors',
    'same_journal',
    'authors_collaboration',
    'collaboration_min_distance',
    'collaboration_mean_distance',
    'writer_in_target_paper',
    'writer_collaborators',
    'writer_collaboration_min_distance',
    'writer_collaboration_mean_distance',
    'source_paper_citations',
    'target_paper_citations',
    'source_number_of_papers_cited',
    'target_number_of_papers_cited',
    'adamic_adar',
    'common_neighbors',
    'jaccard_coefficient',
    'preferential_attachment',
    'max_authors_adamic_adar',
    'max_authors_common_neighbors',
    'max_authors_jaccard_coefficient',
    'max_authors_preferential_attachment',
    'title_cosine_similarity',
    'abstract_cosine_similarity',
    'journal_cosine_similarity',
    'source_title_target_abstract_cosimilarity',
    'source_abstract_target_title_cosimilarity',
    'pagerank_source',
    'pagerank_target'
]

x_train = features_processing(train[selected_columns]).values
x_test = features_processing(test[selected_columns]).values
y_train = train.category.values

# Set model parameters

print('selection and processing done')

print('running cross validation on the chosen model')
seed = 123

model_args = {
    'n_jobs': -1,
    'n_estimators': 100,
    'random_state': seed,
    'criterion': 'entropy',
    'max_features': 'log2'
}

# Cross validation
y_pred, f1_list = cross_validation(RandomForestClassifier, model_args, x_train, y_train)

print('mean f1 during the cross validation of the model: ', np.mean(f1_list))

print('Training final model')
# Train final model and plot features importance

model = RandomForestClassifier(**model_args)
model.fit(x_train, y_train)

print('Plot features importance')
plt.figure(figsize=(20, 10))
plot_features_importance(model.feature_importances_, selected_columns)
plt.show()

print('Generate submission')
# Make the prediction and save it
date = datetime.now().isoformat().split(".")[0]
prediction_name = 'submission.csv'.format(date=date)
y_test_predicted = model.predict(x_test)
y_test_predicted = \
    pd.Series(data=y_test_predicted).rename('category').rename_axis('id', axis='index')
y_test_predicted.to_csv(os.path.join(prediction_name), header=True)

print('This prediction f1 is 0.97023 on the leaderboard')
