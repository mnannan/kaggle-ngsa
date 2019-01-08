from typing import Tuple, Dict

import numpy as np
import pandas as pd
import igraph


class ExtractGraphPapersFeatures:
    """
    This class add the following features:
    - source_paper_citations
    - target_paper_citations
    - source_number_of_papers_cited
    - target_number_of_papers_cited
    - adamic_adar
    - common_neighbors
    - jaccard_coefficient
    - preferential_attachment
    """
    def __init__(self):
        self.graph = None
        self.nodes_mapping = None
        self.df_connected_papers = None

    def fit(self, df: pd.DataFrame):
        """
        Create the papers graph where two papers are connected if one of them as cited the
        other or vice versa
        """
        self.graph, self.nodes_mapping = create_paper_graph(df)
        self.df_connected_papers = df[df.category == 1][['source_id', 'target_id']]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add features related to the papers graph.
        Mandatory columns:
        - source_id
        - target_id
        """

        # Compute neighborhood_based metrics
        metrics = neighborhood_based_metrics(df.source_id, df.target_id, self.graph,
                                             self.nodes_mapping)

        metrics = pd.DataFrame(metrics, index=df.index)

        df = pd.concat([df, metrics], axis='columns')

        paper_citations = \
            self.df_connected_papers.target_id.value_counts().rename('paper_citations')
        df = df.merge(paper_citations.to_frame(), how='left', right_index=True,
                      left_on='source_id')

        df = df.merge(paper_citations.to_frame(), how='left', right_index=True,
                      left_on='target_id', suffixes=('_source', '_target'))
        df = df.rename(columns={
            'paper_citations_source': 'source_paper_citations',
            'paper_citations_target': 'target_paper_citations'
        })

        number_of_papers_cited = self.df_connected_papers.source_id.value_counts().rename(
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


def neighborhood_based_metrics(
        source_id: pd.Series, target_id: pd.Series, graph: igraph.Graph, nodes_mapping: Dict
) -> Dict:
    metrics = {
        'adamic_adar': [],
        'common_neighbors': [],
        'jaccard_coefficient': [],
        'preferential_attachment': []
    }
    for source_id, target_id in zip(source_id.values, target_id.values):
        source_id = nodes_mapping[source_id]
        target_id = nodes_mapping[target_id]
        data = extract_neighborhood_based_metrics(source_id, target_id, graph)
        for metric in metrics:
            metrics[metric].append(data[metric])
    return metrics


def extract_neighborhood_based_metrics(
        source_id: int, target_id: int, graph: igraph.Graph
) -> Dict:
    """
    Given two nodes and a graph compute neighborhood metrics
    :param source_id: Source node
    :param target_id: Target node
    :param graph: graph that contains node
    :return: adamic adar , common neighbors, jaccard coefficient and preferential attachment
    """
    metrics = {}
    source_neighbors = set(graph.neighbors(source_id))
    target_neighbors = set(graph.neighbors(target_id))
    neighbors_intersection = source_neighbors.intersection(target_neighbors)
    neighbors_union = source_neighbors.union(target_neighbors)

    # Extract adamic_adar
    adamic_adar = 0.0
    for neighbor in neighbors_intersection:
        adamic_adar += 1 / np.log(len(graph.neighbors(neighbor)))
    metrics['adamic_adar'] = adamic_adar
    # common_neighbors
    metrics['common_neighbors'] = len(neighbors_intersection)

    # Jaccard coefficient
    if len(neighbors_union) > 0:
        metrics['jaccard_coefficient'] = len(neighbors_intersection) / len(neighbors_union)
    else:
        metrics['jaccard_coefficient'] = 0

    # Preferential attachment
    metrics['preferential_attachment'] = len(source_neighbors) * len(target_neighbors)
    return metrics
