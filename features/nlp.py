
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ExtractCosineSimilarity:
    """
    This class extracts the following features:
    - title_cosine_similarity
    - abstract_cosine_similarity
    - journal_cosine_similarity
    """
    def __init__(self):
        self.papers = None
        self.paper_id_mapping = None
        self.title_tfidf = None
        self.abstract_tfidf = None
        self.journal_tfidf = None
        self.vectorizer = None
        self.journal_vectorizer = None

    def fit(self, df):
        self.papers = df[['source_id', 'source_title', 'source_abstract', 'source_journal']] \
                 .drop_duplicates('source_id').reset_index(drop=True)
        self.paper_id_mapping = {
            source_id: index for index, source_id in list(self.papers.source_id.items())
        }
        self.vectorizer = TfidfVectorizer(analyzer='word', min_df=0, stop_words="english",
                                          ngram_range=(1, 3))
        self.journal_vectorizer = \
            TfidfVectorizer(analyzer='word', min_df=0, stop_words="english",
                            tokenizer=lambda x: x.split('.'), ngram_range=(1, 3))

        self.title_tfidf = self.vectorizer.fit_transform(self.papers.source_title.values)
        self.abstract_tfidf = self.vectorizer.fit_transform(self.papers.source_abstract.values)

        self.journal_tfidf = self.journal_vectorizer.fit_transform(
            self.papers.source_journal.fillna('').values
        )

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:

        source_ids = df.source_id.apply(lambda x: self.paper_id_mapping[x])
        target_ids = df.target_id.apply(lambda x: self.paper_id_mapping[x])

        title_cosimilarity_matrix = cosine_similarity(self.title_tfidf, self.title_tfidf)
        df['title_cosine_similarity'] = title_cosimilarity_matrix[source_ids, target_ids]

        abstract_cosimilarity_matrix = cosine_similarity(self.abstract_tfidf, self.abstract_tfidf)
        df['abstract_cosine_similarity'] = abstract_cosimilarity_matrix[source_ids, target_ids]

        journal_cosimilarity_matrix = cosine_similarity(self.journal_tfidf, self.journal_tfidf)
        df['journal_cosine_similarity'] = journal_cosimilarity_matrix[source_ids, target_ids]
        return df
