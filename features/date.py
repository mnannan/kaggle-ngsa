class PublicationDateDiff:
    """
    This extracts the following features:
    - publication_date_diff
    """

    def fit(self, df):
        pass

    def transform(self, df):
        df['publication_date_diff'] = df.source_publication_date - df.target_publication_date
        return df
