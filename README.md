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

pip install -r requirements.txt


### Data

Download data on https://www.kaggle.com/c/ngsa-w19 and move data in `data` directory.


# Features

## Final
- `title_overlap`: number of common words in both title (on clean titles)
- `publication_date_diff`: Source publication date minus target publication date
- `common_authors`: Number of common authors in source and target authors
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


## Intermediate
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