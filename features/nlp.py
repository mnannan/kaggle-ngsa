
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ExtractCosineSimilarity:
    """
    This class extracts the following features:
    - title_cosine_similarity
    - abstract_cosine_similarity
    - journal_cosine_similarity
    - source_title_target_abstract_cosimilarity
    - source_abstract_target_title_cosimilarity
    """
    def __init__(self):
        self.papers = None
        self.paper_id_mapping = None
        self.title_cosimilarity_matrix = None
        self.abstract_cosimilarity_matrix = None
        self.journal_cosimilarity_matrix = None
        self.title_abstract_cosimilarity_matrix = None

    def fit(self, df):
        self.papers = df[['source_id', 'source_title', 'source_abstract', 'source_journal']] \
            .drop_duplicates('source_id').reset_index(drop=True)
        self.paper_id_mapping = {
            source_id: index for index, source_id in list(self.papers.source_id.items())
        }

        # Vectorizers
        title_vectorizer = TfidfVectorizer(analyzer='word', min_df=0, stop_words="english",
                                           ngram_range=(1, 3))
        abstract_vectorizer = TfidfVectorizer(analyzer='word', min_df=0, stop_words="english",
                                              ngram_range=(1, 3))
        journal_vectorizer = \
            TfidfVectorizer(analyzer='word', min_df=0, stop_words="english",
                            tokenizer=lambda x: x.split('.'), ngram_range=(1, 3))

        title_abstract_vectorizer = TfidfVectorizer(analyzer='word', min_df=0,
                                                    stop_words="english", ngram_range=(1, 3))

        # Embeddings
        title_tfidf = title_vectorizer.fit_transform(
            self.papers.source_title.values
        )
        abstract_tfidf = abstract_vectorizer.fit_transform(
            self.papers.source_abstract.values
        )
        journal_tfidf = journal_vectorizer.fit_transform(
            self.papers.source_journal.fillna('').values
        )
        title_abstract_tfidf = title_abstract_vectorizer.fit(
            pd.concat([self.papers.source_title, self.papers.source_abstract]).values
        )

        # Cosimilarities
        self.title_cosimilarity_matrix = cosine_similarity(title_tfidf, title_tfidf)

        self.abstract_cosimilarity_matrix = cosine_similarity(abstract_tfidf, abstract_tfidf)

        self.journal_cosimilarity_matrix = cosine_similarity(journal_tfidf, journal_tfidf)
        self.title_abstract_cosimilarity_matrix = \
            cosine_similarity(
                title_abstract_tfidf.transform(self.papers.source_title.values),
                title_abstract_tfidf.transform(self.papers.source_abstract.values)
            )

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:

        source_ids = df.source_id.apply(lambda x: self.paper_id_mapping[x])
        target_ids = df.target_id.apply(lambda x: self.paper_id_mapping[x])

        df['title_cosine_similarity'] = self.title_cosimilarity_matrix[source_ids, target_ids]

        df['abstract_cosine_similarity'] = \
            self.abstract_cosimilarity_matrix[source_ids, target_ids]

        df['journal_cosine_similarity'] = self.journal_cosimilarity_matrix[source_ids, target_ids]

        df['source_title_target_abstract_cosimilarity'] = \
            self.title_abstract_cosimilarity_matrix[source_ids, target_ids]

        df['source_abstract_target_title_cosimilarity'] = \
            self.title_abstract_cosimilarity_matrix[target_ids, source_ids]
        return df
