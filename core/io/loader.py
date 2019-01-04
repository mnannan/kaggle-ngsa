import os

import pandas as pd


def load_data(dataset: str, data_dir: str ='./data/') -> pd.DataFrame:
    """
    Load data with right columns name
    :param dataset: dataset we want to merge need to be in
    :param data_dir: directory that contains the data
    e.g data_dir=/Users/mnannan/kaggle-ngsa/data/
    It's optional because data is supposed to be in data directory
    :return:
    """
    read_args = {
        'node_information':{
            'names': ['id', 'publication_date', 'title', 'authors', 'journal', 'abstract'],
            'index_col': 'id'
        },
        'train': {
            'names': ['source_id', 'target_id','category'],
            'sep': ' ',
        },
        'test':{
            'names':['source_id', 'target_id'],
            'sep': ' ',
        }
    }
    filenames = {
        'node_information': 'node_information.csv',
        'train': 'training_set.txt',
        'test': 'testing_set.txt'
    }
    if dataset in read_args:
        try:
            file_path = os.path.join(data_dir, filenames[dataset])
            return pd.read_csv(file_path, header=None, **read_args[dataset])
        except FileNotFoundError as e:
            raise FileNotFoundError('You need to set data_dir parameter to the right directory')
    else:
        raise RuntimeError(f'{dataset} does not exist')


def merge_node_information(dataset: pd.DataFrame, node_information: pd.DataFrame) -> pd.DataFrame:
    """
    Merge node information with training or testing dataset
    It adds source_ or target_ in front of the field
    """
    for column in node_information.columns:
        for origin in ['source', 'target']:
            serie = node_information[column].rename(f'{origin}_{column}').to_frame()
            dataset = dataset.merge(serie, left_on=f'{origin}_id', right_index=True)
    return dataset


def get_data_with_node_information(dataset: pd.DataFrame, data_dir='./data/') -> pd.DataFrame:
    """
    Same as load_data but return the dataset with node_information
    e.g: get_data_with_node_information('train') will return train dataset with node_information
    columns
    """
    df = load_data(dataset, data_dir)
    node_information = load_data('node_information', data_dir)
    return merge_node_information(df, node_information)
