# review dataset
This repository contains analysis on relationship between liguistic features and acceptance results for academic papers. The dataset is constructed from revision history of all 9895 papers submitted to ICLR through the OpenReview from 2018 to 2021, discarding any papers that have only one draft. In addition, the paper level data also includes review data such as review scores, rebuttals, and meta_reviews. 

## paper level data schema
The paragraph level data contains tokenized text of the abstract, introduction and the conclusion section of the first draft of a paper, as well as computed linguistic features (correctness and readability). The data is stored in data/paper_features_2.tsv. 
| Column name   | Description                                                         |
| ------------- | -----------------------------------------------------------         |
| paperid         | OpenReview paper id for the extracted text.                     |
| text         | List of tokenized sentences in the abstract, introduction and conclusion section           |
| correct         | Correctness score (computed as the fraction of grammatically correct sentences).               |
| readbility | Readability score (computed as the average of flesch_kincaid_grade_level and coleman_liau_index). The lower the readability score, the harder to parse the text.                         |
| Acceptance    | The acceptance result of the paper. Oral/Spotlight: 0; Poster/Workshop: 1; Reject/Withdrawn: 2  |



## sentence level data schema
The sentence level data is in data/clean_edits_2.tsv. 
| Column name   | Description                                                         |
| ------------- | -----------------------------------------------------------         |
| Paper         | OpenReview paper id for the extracted sentence.                     |
| Sen_1         | Sentence before revision (usually in the first draft)               |
| Sen_2         | Sentence after revision (usually in the first draft)                |
| Context_before| The preceding context sentence to Sen_1.                            |
| Context_after | The succeding context sentence to Sen_1.                            |
| Acceptance    | The acceptance result of the paper. Oral/Spotlight: 0; Poster/Workshop: 1; Reject/Withdrawn: 2  |










## Extraction process
### Get a list of paper ids as well as their revisions urls
python spider.py 

### Download the pdfs using the urls 
python downloader.py

### Compile the dataset and save to json
python dataset.py

### Replicate the training procedure for paper result prediction
python model.py --dataset paper --num_epochs 10
