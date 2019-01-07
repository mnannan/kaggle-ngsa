from typing import List


import pandas as pd


class FeaturesExtractor:
    """
    Class that given Extractors extracts features from a pandas dataframe

    e.g:
    features_extractor = FeaturesExtractor([TitleOverlap(), CommonAuthors()])

    Every extractor fed to FeaturesExtractor has to have a fit and a transform method
    """
    def __init__(self, pipeline: List):
        self.pipeline = pipeline

    def transform(self, df) -> pd.DataFrame:
        """ Use this for test set"""
        for task in self.pipeline:
            df = task.transform(df)
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Use this for train set"""
        for task in self.pipeline:
            task.fit(df)
            df = task.transform(df)
        return df
