# Kaggle NGSA

https://www.kaggle.com/c/ngsa-w19

This projects aims to solve kaggle competition about predicting whether one paper will quote 
another paper given:
-   A graph of papers
-   For each paper:
     - Title
     - Publication date
     - Authors
     - Abstract
     - Journal 
        
# Installation

### Requirements 

To install requirements run 
```
pip install -r requirements.txt
```

### Data

Download data on [kaggle competition](https://www.kaggle.com/c/ngsa-w19) and move data in `data` 
directory.


# Features

## Training features
- `title_overlap`: number of common words in both titles (on clean titles)
- `publication_date_diff`: Source publication date minus target publication date
- `common_authors`: Number of common authors in source and in target
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
- `source_target_common_neighbors`: Number of common neighbors between target and source in 
papers graph.
- `source_paper_citations`: Number of papers that have cited source paper.
- `target_paper_citations`: Number of papers that have cited target paper.
- `source_number_of_papers_cited`: Number of papers that source has cited.
- `target_number_of_papers_cited`: Number of papers that target has cited.
- `same_journal`: Boolean that tells whether source and target belong to the same journal. 

## Intermediate features
- `source_authors_list`: list of authors extracted with regex
- `target_authors_list`: same as `source_authors_list`
- `collaboration_paths`: distance between source authors and target authors in the collaboration 
graph
- `writer_collaboration_distance`: distance between source writer and target authors in the 
collaboration graph


# Graph

## Collaboration graph

This graph has been built with authors of each paper. 
Nodes are authors and *u* and *v* are connected if *u* and *v* have co writen a paper.

## Papers graph

This graph has been built with papers relations. It's a non directed graph.
Nodes are papers and u and v are connected if *u* has cited *v* or *v* has cited *u*
