from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
import igraph
from sklearn.utils.extmath import cartesian


class CollaborationDistance:
    """
    This extracts the following features:
    - collaboration_distance
    - writer_collaboration_distance
    """
    def __init__(self):
        self.collaboration_graph = None
        self.authors_mapping = None
        self.distances_matrix = None

    def fit(self, df):
        pass

    def transform(self, df):

        self.collaboration_graph, self.authors_mapping = create_collaboration_graph(df)
        self.distances_matrix = extract_distances_matrix(self.collaboration_graph)
        authors_list = \
            df.source_authors_list.apply(lambda x: [x] if x is not None else None) + \
            df.target_authors_list.apply(lambda x: [x] if x is not None else None)
        df['collaboration_distance'] = authors_list.apply(
            lambda x: extract_distances_series(x, self.distances_matrix, self.authors_mapping)
        )
        # We just keep the author of the paper and target authors
        writer_target_authors_list = \
            df.source_authors_list.apply(
                lambda x: [[x[0]]] if (isinstance(x, list) and len(x) > 0) else None
            ) + \
            df.target_authors_list.apply(lambda x: [x] if x is not None else None)
        df['writer_collaboration_distance'] = writer_target_authors_list.apply(
            lambda x: extract_distances_series(x, self.distances_matrix, self.authors_mapping)
        )
        return df


class CollaborationFeatures:
    """
    This extracts the following features:
    - common_authors
    - authors_collaboration
    - collaboration_min_distance
    - collaboration_mean_distance
    - writer_in_target_paper
    - writer_collaborators
    - writer_collaboration_min_distance
    - writer_collaboration_mean_distance
    """
    def fit(self, df):
        pass

    def transform(self, df):

        df['common_authors'] = df.collaboration_distance.apply(
            lambda x: number_of_occurrences(x, 0)
        )
        df['authors_collaboration'] = \
            df.collaboration_distance.apply(lambda x: number_of_occurrences(x, 1))
        df['collaboration_min_distance'] = \
            df.collaboration_distance.apply(lambda x: np.min(x) if x is not None else np.inf)
        df['collaboration_mean_distance'] = \
            df.collaboration_distance.apply(lambda x: np.mean(x) if x is not None else np.inf)

        df['writer_in_target_paper'] = \
            df.writer_collaboration_distance.apply(lambda x: isinstance(x, list) and 0 in x)
        df['writer_collaborators'] = \
            df.writer_collaboration_distance.apply(lambda x: number_of_occurrences(x, 1))
        df['writer_collaboration_min_distance'] = \
            df.writer_collaboration_distance.apply(
                lambda x: np.min(x) if x is not None else np.inf
            )
        df['writer_collaboration_mean_distance'] = \
            df.writer_collaboration_distance.apply(
                lambda x: np.mean(x) if x is not None else np.inf
            )

        return df


def extract_source_target_authors_list(
        df: pd.DataFrame
) -> Tuple[List[Tuple[int, str]], List[Tuple[int, str]]]:
    """
    Needed columns:
    - source_authors_list
    - source_id
    - target_id
    - target_authors_list
    :param df:
    :return:
    list of tuple that contains (paper_id, author_name)

    """
    source_authors_list = []
    target_authors_list = []
    source_df = df[['source_id', 'source_authors_list']].drop_duplicates('source_id')
    for source_id, authors in zip(source_df.source_id.values,
                                  source_df.source_authors_list.values):
        if authors:
            for author in authors:
                source_authors_list.append((source_id, author))
    target_df = df[['target_id', 'target_authors_list']].drop_duplicates('target_id')
    for target_id, authors in zip(target_df.target_id.values,
                                  target_df.target_authors_list.values):
        if authors:
            for author in authors:
                target_authors_list.append((target_id, author))
    return source_authors_list, target_authors_list


def merge_authors(author1: pd.Series, author2: pd.Series) -> pd.Series:
    """
    This allow us to have an unique identifier for tuples that does not care about the order.
    merge_authors(a,b) = merge_authors(b,a)
    """
    concat_authors = author1.astype('str') + ',' + author2.astype('str')
    return concat_authors.apply(lambda x: ''.join(sorted(x.split(','))))


def create_collaboration_graph(df: pd.DataFrame) -> Tuple[igraph.Graph, Dict]:
    """
    This create the collaboration graph whose:
    - Nodes are authors
    - u and v are connected if u and v have co written a paper


    Mandatory columns:
    - source_authors_list
    - source_id
    - target_id
    - target_authors_list
    :param df:
    :return:
    """
    source_authors_list, _ = extract_source_target_authors_list(df)
    source_authors = pd.DataFrame(source_authors_list)
    source_authors = source_authors.rename({0: 'paper_id', 1: 'author'}, axis='columns')

    authors_mapping = {value: key for key, value in
                       source_authors.author.drop_duplicates().reset_index(
                           drop=True).items()}

    collaboration = source_authors.merge(source_authors, left_on='paper_id',
                                         right_on='paper_id', suffixes=('_1', '_2'))
    collaboration = collaboration[collaboration.author_1 != collaboration.author_2]
    collaboration['merge_authors'] = merge_authors(collaboration.author_1, collaboration.author_2)
    collaboration = collaboration.sort_values(['paper_id', 'author_1', 'author_2'])\
                                 .drop_duplicates(['paper_id', 'merge_authors'])

    collaboration_edges = []
    for author_1, author_2 in zip(collaboration.author_1, collaboration.author_2):
        collaboration_edges.append((authors_mapping[author_1], authors_mapping[author_2]))

    return igraph.Graph(collaboration_edges), authors_mapping


def extract_distances_matrix(graph: igraph.Graph) -> np.matrix:
    """Extract distances matrix from igraph.Graph"""
    return np.matrix(graph.shortest_paths())


def extract_distances_series(
        nodes_list: pd.Series, shortest_paths_matrix: np.matrix, nodes_mapping: Dict
) -> pd.Series:
    """
    Given a list of two list of nodes, return shortest paths between all the pair of nodes from
    each list
    e.g Given [[node_1, node_2],[node_3]] this will output shortest paths between 1 and 3 and 2
    and 3
    :param nodes_list:
    :param shortest_paths_matrix: matrix generated by extract_shortest_paths_matrix
    :param nodes_mapping: dict
    {
        node_name found in node_list : id used to encode the node in the matrix
    }
    e.g
    { node1: 1: node2: 2, node3: 3}
    :return:
    """
    if isinstance(nodes_list, list) and len(nodes_list) == 2 and nodes_list[0] and nodes_list[1]:
        mapped_nodes_1 = np.array([nodes_mapping[node] for node in nodes_list[0]])
        mapped_nodes_2 = np.array([nodes_mapping[node] for node in nodes_list[1]])
        c = cartesian((mapped_nodes_1, mapped_nodes_2))
        return shortest_paths_matrix[c[:, 0], c[:, 1]].tolist()[0]


def number_of_occurrences(l: List[int], x: int) -> int:
    """ Given a list of int, returns number of occurences of x"""
    cpt = 0
    if isinstance(l, list):
        for element in l:
            if element == x:
                cpt += 1
    return cpt
