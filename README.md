# Math Problem Multi-Label Classification

A multi-label text classifier that automatically tags mathematics problems with one or more of four core areas: **Algebra**, **Geometry**, **Number Theory**, and **Combinatorics**.

## Overview

- **Input:** Raw mathematical text (LaTeX, forum posts).
- **Output:** Multi-hot vector of shape `[n, 4]` (classes are non-mutually exclusive).
- **Dataset:** Sourced from [AoPS Crawler](https://github.com/hoang2008an/Aops_Crawler).
- **Tokenization:** SentencePiece (SPM) model trained on a 190MB text corpus extracted directly from the dataset.

## Models

| Model | Representation / Architecture | Notebook |
|---|---|:---:|
| **XGBoost** | TF-IDF features (n-gram baseline) | — |
| **LSTM** | Custom PyTorch LSTM with SentencePiece embeddings | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/11Z1hxvwN6LVG1mC6Hx0y6iDFDRceNQR4?usp=sharing) |
| **MathBERT** | Fine-tuned [`tbs17/MathBERT`](https://huggingface.co/tbs17/MathBERT) via Hugging Face `Trainer` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1SiXmUtuLf4ymV4njTwA__kA7EkJH1FHx?usp=sharing) |

## Evaluation & Results

Evaluated across per-label accuracy, sample-averaged Jaccard similarity, and F1 scores.

| Metric | XGBoost | LSTM | MathBERT |
|---|:---:|:---:|:---:|
| **Accuracy** | | | |
| ├ Algebra | 0.921 | 0.922 | **0.927** |
| ├ Number Theory | 0.921 | 0.915 | **0.926** |
| ├ Geometry | 0.956 | 0.956 | **0.957** |
| └ Combinatorics | 0.928 | 0.926 | **0.936** |
| **Jaccard (samples)** | 0.847 | 0.861 | **0.875** |
| **F1 (micro)** | 0.873 | 0.872 | **0.884** |
| **F1 (macro)** | 0.860 | 0.858 | **0.873** |
| **Precision (micro)** | 0.901 | 0.888 | **0.902** |
| **Recall (micro)** | 0.848 | 0.857 | **0.867** |

### Key Takeaway

All three models show nearly identical performance. Even with specialized pretraining, MathBERT fails to pull ahead of XGBoost or LSTM, indicating a data bottleneck rather than a modeling limitation. A quick error analysis confirmed frequent mislabeling. 