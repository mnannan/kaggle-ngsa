from typing import List, Optional
import re


class ExtractAuthorsList:

    def fit(self, df):
        pass

    def transform(self, df):
        df['source_authors_list'] = df.source_authors.apply(lambda x: extract_authors_list(x))
        df['target_authors_list'] = df.target_authors.apply(lambda x: extract_authors_list(x))
        return df


def extract_authors_list(authors: str) -> Optional[List[str]]:
    stopwords = ['alex', 't', 'latex', 'pages', 'jr', 's']
    if isinstance(authors, str):

        # Remove string between parenthesis
        string = re.sub(r'\([^\)]+[\)$]', '', authors)
        # Remove bad parenthesis
        string = re.sub(r'\([^\)]*$', '', string)
        # Remove multiple spaces
        string = re.sub(r'\s', '', string)
        # Extract authors and sanitize strings

        authors = []
        for author in string.split(','):
            author = author.lower()
            author = re.sub(r'[\W\d]', '', author).lower()
            if len(author) >= 2 and author not in stopwords:
                authors.append(author)
        return authors
