import pandas as pd


class ExtractJournalFeatures:
    """
    Add the following columns:
    - same_journal
    """
    def fit(self, df):
        pass

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add a boolean that tells you whether target and source belong to the same journal
        Mandatory columns :
        - source_journal
        - target_journal
        """
        df['same_journal'] = (df.source_journal == df.target_journal)
        return df
