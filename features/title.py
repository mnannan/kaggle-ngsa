from features.string import overlap, clean_string


class CleanTitle:
    """
    This class clean source_title and target_title columns
    """
    def fit(self, df):
        pass

    def transform(self, df):
        df['source_title'] = df.source_title.apply(clean_string)
        df['target_title'] = df.target_title.apply(clean_string)
        return df


class TitleOverlapping:
    """
    This extracts title overlapping
    """
    def fit(self, df):
        pass

    def transform(self, df):

        df['title_overlap'] = df.apply(
            lambda x: overlap(x.source_title, x.target_title), axis='columns'
        )
        return df
