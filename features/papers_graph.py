from typing import Tuple, Dict

import pandas as pd
import igraph


class ExtractGraphPapersFeatures:
    """
    This class add the following features:
    - source_target_common_neighbors
    - source_paper_citations
    - target_paper_citations
    - source_number_of_papers_cited
    - target_number_of_papers_cited
    """
    def __init__(self):
        self.graph = None
        self.nodes_mapping = None

    def fit(self, df: pd.DataFrame):
        """
        Create the papers graph where two papers are connected if one of them as cited the
        other or vice versa
        """
        self.graph, self.nodes_mapping = create_paper_graph(df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add features related to the papers graph.
        Mandatory columns:
        - source_id
        - target_id
        """
        neighbors = {}
        for i in range(self.graph.vcount()):
            neighbors[i] = set(self.graph.neighbors(i))

        common_neighbors = []
        for source_id, target_id in zip(df.source_id.values, df.target_id.values):
            source_id = self.nodes_mapping[source_id]
            target_id = self.nodes_mapping[target_id]
            common_neighbors.append(len(neighbors[source_id].intersection(neighbors[target_id])))

        df['source_target_common_neighbors'] = pd.Series(common_neighbors, index=df.index)

        paper_citations = df[df.category == 1].target_id.value_counts().rename(
            'paper_citations')
        df = df.merge(paper_citations.to_frame(), how='left', right_index=True,
                      left_on='source_id')

        df = df.merge(paper_citations.to_frame(), how='left', right_index=True,
                      left_on='target_id', suffixes=('_source', '_target'))
        df = df.rename(columns={
            'paper_citations_source': 'source_paper_citations',
            'paper_citations_target': 'target_paper_citations'
        })

        number_of_papers_cited = df[df.category == 1].source_id.value_counts().rename(
            'number_of_papers_cited')
        df = df.merge(number_of_papers_cited.to_frame(), how='left', right_index=True,
                      left_on='source_id')

        df = df.merge(number_of_papers_cited.to_frame(), how='left', right_index=True,
                      left_on='target_id', suffixes=('_source', '_target'))

        df = df.rename(columns={
            'number_of_papers_cited_source': 'source_number_of_papers_cited',
            'number_of_papers_cited_target': 'target_number_of_papers_cited'
        })

        return df


def create_paper_graph(df: pd.DataFrame) -> Tuple[igraph.Graph, Dict]:
    """
    Mandatory columns:
    -   source_id
    -   target_id
    This returns the papers graph whose:
    -   Nodes are articles
    -   u and v are connected if u has quoted v
    This also return nodes mapping dict {paper_id: id in the graph}
    """
    nodes_mapping = df.source_id.drop_duplicates().reset_index(drop=True).to_dict()
    nodes_mapping = {value: key for key, value in nodes_mapping.items()}
    edges = []
    connected_papers = df[df.category == 1]
    for source_id, target_id in zip(connected_papers.source_id, connected_papers.target_id):
        edges.append((nodes_mapping[source_id], nodes_mapping[target_id]))
    graph = igraph.Graph(edges)
    return graph, nodes_mapping
