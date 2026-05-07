<div align="center">

<h1>NER Twitter Project</h1>

<p>
  <strong>Named Entity Recognition for noisy Twitter text using HMM, CRF, and BiLSTM-CRF models</strong>
</p>

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.13-blue">
  <img alt="Notebook" src="https://img.shields.io/badge/Jupyter-Notebook-orange">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-BiLSTM--CRF-red">
  <img alt="Task" src="https://img.shields.io/badge/Task-NER-purple">
  <img alt="Models" src="https://img.shields.io/badge/Models-HMM%20%7C%20CRF%20%7C%20BiLSTM--CRF-brightgreen">
</p>

<p>
  <a href="#motivation">Motivation</a> |
  <a href="#methods">Methods</a> |
  <a href="#setup">Setup</a> |
  <a href="#recorded-results">Results</a> |
  <a href="#recommendations">Recommendations</a>
</p>

</div>

---

## Project Snapshot

| Item | Details |
| --- | --- |
| Domain | Twitter Named Entity Recognition |
| Entity types | `PER`, `LOC`, `ORG`, `MISC` |
| Code language | Python, Jupyter Notebook |
| Models used | HMM, CRF, BiLSTM-CRF |
| Best final model | Contextual CRF (`A2`) |
| Best official test F1 | `0.6858` |
| Main notebook | `NER_TC/NER Project/NER/notebook/NER Project.ipynb` |
| Embeddings | GloVe Twitter 27B, 100d |

## Overview

This repository contains a Jupyter notebook for named entity recognition (NER) on tweets. The project compares three approaches for extracting entities from short, noisy social-media text:

- Method A: feature-based Conditional Random Field (CRF)
- Method B: Hidden Markov Model (HMM)
- Method C: BiLSTM-CRF in PyTorch with word and character features

The notebook preprocesses tweet text, converts entity annotations into BIO tags, performs 5-fold cross-validation, and evaluates the selected models on the official test set.

## Motivation

Named Entity Recognition turns unstructured text into structured information by identifying entities such as people, locations, organizations, and miscellaneous named entities. This is useful for social media monitoring, news analysis, customer feedback analysis, and automatic content tagging.

Tweets are a difficult NER domain because they are short, informal, noisy, and often contain mentions, hashtags, URLs, abbreviations, spelling variation, and inconsistent capitalization. This project therefore compares classical, feature-based, and neural sequence-labelling methods to test whether more complex models actually improve performance on a small Twitter NER dataset.

The project also reflects an evolution in NLP modelling thinking: it starts from a basic statistical sequence model (HMM), moves through a bridge of feature-based machine learning (CRF with handcrafted contextual features), and ends with a deep learning approach (BiLSTM-CRF with word and character representations). This progression makes the comparison useful not only as a performance benchmark, but also as a practical study of how NLP methods become more expressive as their representations and learning mechanisms become more advanced.

## Research Questions

This project focuses on three questions:

1. To what extent can named entities in tweets be automatically extracted and classified into `PER`, `LOC`, `ORG`, and `MISC` categories?
2. Which modelling approach performs best on this dataset: HMM, CRF, or BiLSTM-CRF?
3. Does adding richer contextual or character-level information improve NER performance on noisy tweets?

## Repository Structure

```text
.
|-- README.md
|-- .gitignore
`-- NER_TC/
    `-- NER Project/
        `-- NER/
            |-- data/
            |   |-- TweetsTrainset.txt
            |   |-- TweetsTestset.txt
            |   `-- TweetsTestGroundTruth.txt
            `-- notebook/
                `-- NER Project.ipynb
```

## Data

The dataset is stored in tab-separated text files under `NER_TC/NER Project/NER/data/`.

| File | Rows | Description |
| --- | ---: | --- |
| `TweetsTrainset.txt` | 2,815 | Training tweets with tweet ID, entity annotations, and tweet text |
| `TweetsTestset.txt` | 1,450 | Official test tweets without inline labels |
| `TweetsTestGroundTruth.txt` | 1,450 | Ground-truth entity annotations for the test set |

Training annotations use a semicolon-separated format such as:

```text
ORG/WikiLeaks;LOC/Southern Ocean;    tweet text...
```

The notebook maps these annotations into BIO tags such as `B-PER`, `I-PER`, `B-ORG`, `B-LOC`, `B-MISC`, and `O`.

Important data characteristics:

- Tweets include noisy social-media artifacts such as mentions, hashtags, and URLs.
- Multi-word entities require boundary-aware tagging, so BIO tags are used.
- The dataset is highly imbalanced because most tokens are labelled `O`.
- Rare and unseen entities are common, which makes generalization difficult.

## Methods

### Method A: CRF

The CRF approach uses manually engineered token features. Two variants are evaluated:

- `A1`: basic lexical, shape, suffix, prefix, POS, mention, hashtag, and URL features
- `A2`: contextual features with neighboring token information

### Method B: HMM

The HMM approach models BIO tag transitions and token emissions with:

- Laplace smoothing
- rare and unknown-word signatures
- Viterbi decoding
- BIO transition constraints

### Method C: BiLSTM-CRF

The neural approach is implemented in PyTorch. Two variants are evaluated:

- `C1`: word embeddings only
- `C2`: word embeddings plus character-level features

The model uses 100-dimensional GloVe Twitter embeddings and character-level features to better handle slang, abbreviations, misspellings, and out-of-vocabulary words.

## Preprocessing

The notebook applies the following preprocessing steps:

- Convert entity annotations into BIO sequence labels.
- Replace Twitter-specific artifacts with normalized tokens:
  - `@mentions` -> `_MENTION_`
  - `#hashtags` -> `_HASHTAG_`
  - URLs -> `_URL_`
- Tokenize tweets while preserving entity boundaries.
- Build token, character, and tag vocabularies for the neural model.
- Use stratified 5-fold cross-validation to reduce dependence on a single train/validation split.

## Setup

The notebook metadata shows it was run with Python 3.13.12. A fresh environment can be prepared with:

```bash
cd "NER_TC/NER Project/NER"
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r ../../../requirement.txt
```

### GloVe Twitter Embeddings

Method C expects the GloVe Twitter 27B vectors. The notebook looks for the 100-dimensional file at:

```text
glove.twitter.27B/glove.twitter.27B.100d.txt
```

Download the Twitter vectors from the official Stanford GloVe page:

<https://nlp.stanford.edu/projects/glove/>

The specific file listed there is `glove.twitter.27B.zip` with Twitter vectors trained on 2B tweets and 27B tokens. From inside `NER_TC/NER Project/NER`, you can install it manually with:

```bash
curl -L -O https://nlp.stanford.edu/data/glove.twitter.27B.zip
unzip glove.twitter.27B.zip -d glove.twitter.27B
```

After extraction, confirm that this file exists:

```text
NER_TC/NER Project/NER/glove.twitter.27B/glove.twitter.27B.100d.txt
```

The notebook also includes fallback logic that attempts to download the archive automatically. Manual setup is recommended because the archive is large, around 1.42 GB, and should not be committed to Git.

Then start Jupyter from the same directory:

```bash
jupyter lab
```

Open:

```text
notebook/NER Project.ipynb
```

The notebook uses relative paths such as `data/TweetsTrainset.txt`, so make sure the working directory is `NER_TC/NER Project/NER` when running it.

## Running the Notebook

Run the notebook cells from top to bottom. The workflow is:

1. Load and inspect the tweet datasets.
2. Tokenize tweets and convert entity annotations to BIO tags.
3. Create stratified 5-fold train/validation splits.
4. Train and evaluate Method A CRF variants.
5. Train and evaluate Method B HMM.
6. Train and evaluate Method C BiLSTM-CRF variants.
7. Retrain selected models on the full training set.
8. Evaluate final models on the official test set.
9. Plot final F1-score comparison and confusion matrices.

Method C can take noticeably longer than the CRF and HMM baselines, especially if the GloVe vectors need to be downloaded.

## Evaluation

The project uses micro-average precision, recall, and F1-score because entity classes are imbalanced and most tokens are outside entities. Final evaluation uses strict IOB2 entity scoring, meaning an entity is only correct when both its boundary and type match the ground truth.

The notebook also uses confusion matrices to inspect errors between entity types such as `PER`, `LOC`, `ORG`, and `MISC`.

## Recorded Results

The current notebook output records the following official test-set F1 scores:

| Method | Selected Variant | Test F1 |
| --- | --- | ---: |
| Method A | CRF `A2` | 0.6858 |
| Method B | HMM | 0.5564 |
| Method C | BiLSTM-CRF `C2` | 0.6479 |

The best recorded official test score is Method A, the contextual CRF variant.

Cross-validation summaries recorded in the notebook:

| Method | Variant | Mean F1 |
| --- | --- | ---: |
| Method A | `A2` contextual CRF | 0.6652 +/- 0.0216 |
| Method C | `C1` word only | 0.6996 +/- 0.0148 |
| Method C | `C2` word + char | 0.7162 +/- 0.0145 |

## Conclusions

The contextual CRF model `A2` is the strongest final model for this specific dataset. Although BiLSTM-CRF `C2` achieved the best cross-validation score, it generalized less well on the independent test set. The most likely reasons are the small training set, noisy tweet language, and unseen entities in the test data.

The results also show that model complexity alone does not guarantee better performance. Feature-engineered CRF models can be more reliable than neural models when labelled data is limited.

Entity-level performance shows that `PER` is the easiest category, while `MISC`, `LOC`, and `ORG` are harder because they are less frequent and more heterogeneous.

## Recommendations

- Use the contextual CRF model `A2` as the current production candidate for a lightweight and interpretable Twitter NER system.
- Prefer CRF when labelled data is limited and test data may contain many unseen names.
- Expand the labelled dataset, especially for underrepresented categories such as `MISC`, `LOC`, and `ORG`.
- Add manual error analysis to determine whether most mistakes come from boundary detection, entity type confusion, or unseen entities.
- Test transformer-based models such as BERT or RoBERTa in future work.
- Explore data augmentation and domain-specific pretraining to improve robustness on noisy tweets.
- Keep using Twitter-specific embeddings or other domain-specific representations when training neural models.

## Limitations

- The training set is small for deep learning, with only 2,815 tweets.
- Replacing mentions, hashtags, and URLs reduces noise but can remove useful semantic context.
- The system still struggles with completely unseen entities and out-of-vocabulary names.
- The neural model may overfit or fail to fully converge because of the limited dataset size.

## Notes

- Generated model artifacts and downloaded embeddings are not required to inspect the repository.
- The notebook installs a few packages inline with `%pip install`, but installing dependencies before running the notebook is recommended.
- The project currently centers on the notebook workflow rather than a reusable Python package or command-line interface.
