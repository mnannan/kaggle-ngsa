# Kaggle NGSA

https://www.kaggle.com/c/ngsa-w19

This projects aims to solve kaggle competition about predicting whether one paper will cite
another paper given:
-   A graph of papers
-   For each paper:
     - Title
     - Publication date
     - Authors
     - Abstract
     - Journal 
      
The python file `main.py` file generates a submission that has a f1 of 0.97023 on the leaderboard
# Installation

### Requirements 

To install requirements run 
```
pip install -r requirements.txt
```

### Data

Download data on [the kaggle competition](https://www.kaggle.com/c/ngsa-w19) and move data in `data` 
directory.


# Running the submission

To generate the submission you just need to run the following command from the project root
```
python main.py
```
This will output a csv `submission.csv`, print cross validation results and plot features importance


# Project structure

## Directories

### features/
Contains all the .py files to generate the features

### models/
Contains a cross validation util

### plot/
Contains utils used during data analysis

### core/
Utils to load datasets


## Features extractors

We used scikit pipeline style to design our features extractors which means that they have two 
methods one `fit` method in order to build graph for example and `transform` method that given a 
pandas dataframe returns the input pandas dataframe with the extracted features.
# Features

## Training features

### Basic features
- `title_overlap`: number of common words in both titles (on clean titles)
- `publication_date_diff`: Source publication date minus target publication date
- `common_authors`: Number of common authors in source and in target
### Graph Features
- `authors_collaboration`: Number of pairs of authors from source and target papers 
that have collaborated.
- `collaboration_min_distance`: Minimum distance in the collaboration graph between authors from 
source paper and authors from target paper.
- `collaboration_mean_distance`: Mean distance in the collaboration graph between authors from 
source paper and authors from target paper.
- `writer_in_target_paper`: Boolean that tells us whether the writer (First author) of the source 
paper in 
- `writer_collaborators`: Number of collaborators of the writer
- `writer_collaboration_min_distance`: Minimum distance between writer and collaborators.
- `writer_collaboration_mean_distance`: Mean distance between writer and collaborators.
- `common_neighbors`: Number of common neighbors between target and source in 
papers graph.
- `source_paper_citations`: Number of papers that have cited source paper.
- `target_paper_citations`: Number of papers that have cited target paper.
- `source_number_of_papers_cited`: Number of papers that source has cited.
- `target_number_of_pape`rs_cited`: Number of papers that target has cited.
- `same_journal`: Boolean that tells whether source and target belong to the same journal. 
- `adamic_adar`: Adamic adar index between source and target papers in papers graph
- `jaccard_coefficient`:Jaccard coefficient between source and target papers in papers
 graph
- `preferential_attachment`: Preferential attachment between target and source papers in papers graph
- `max_authors_adamic_adar`: Maximum adamic adar between authors from source paper and target 
paper in authors graph
- `max_authors_common_neighbors`:Maximum number of common neighbors between authors from 
source paper and target 
paper in authors graph
- `max_authors_jaccard_coefficient`:Maximum Jaccard coefficient between authors from source paper 
and target paper in authors graph
- `max_authors_preferential_attachment`:Maximum Preferential attachment between authors from source 
paper and target paper in authors graph
- `pagerank_source`: Pagerank of the source paper
- `pagerank_target`: Pagerank of the target paper

### NLP Features
- `title_cosine_similarity`: cosine similarity between the tfidf of the target title and the 
source title. 
- `abstract_cosine_similarity`: cosine similarity between the tfidf of the target abstract and the 
source abstract. 
- `journal_cosine_similarity`: cosine similarity between the tfidf of the target journal and the 
source journal. 
- `source_abstract_target_title_cosimilarity`: cosine similarity of the tfidf of the source 
abstract and the target title.
- `source_title_target_abstract_cosimilarity`: cosine similarity of the tfidf of the source 
title and the target abstract.

## Intermediate features
- `source_authors_list`: list of authors extracted with regex
- `target_authors_list`: same as `source_authors_list`
- `collaboration_paths`: distance between source authors and target authors in the collaboration 
graph
- `writer_collaboration_distance`: distance between source writer and target authors in the 
collaboration graph

# Graph built

## Collaboration graph

This graph has been built with authors of each paper. 
Nodes are authors and *u* and *v* are connected if *u* and *v* have co writen a paper.

## Papers graph

This graph has been built with papers relations. It's a non directed graph.
Nodes are papers and u and v are connected if *u* has cited *v* or *v* has cited *u*
