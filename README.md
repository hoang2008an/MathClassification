# Math Question Multi-Label Classification

A multi-label classifier that tags math questions/threads with one or more of four topics: **Algebra**, **Geometry**, **Number Theory**, **Combinatorics**.

## Problem Setup

- **Input (X):** raw text, tokenized manually before feeding into the models
- **Output (Y):** multi-hot label vector of shape `[n, 4]`
- **Task type:** multi-label classification (non-mutually-exclusive)

## Preprocessing
 
The dataset used for training/evaluation is sourced from [this repo](https://github.com/hoang2008an/Aops_Crawler).
 
Tokenization uses a custom vocabulary trained with **SentencePiece (SPM)** on a 190MB text corpus from the dataset.
## Models Compared

| Model | Description |
|---|---|
| **XGBoost** | Gradient-boosted trees on sparse text features |
| **LSTM** | Custom PyTorch LSTM handling variable-length sequences, trained on Colab |
| **MathBERT** | [MathBERT](https://huggingface.co/tbs17/MathBERT) fine-tuned via Hugging Face `Trainer`, trained on Colab |

- LSTM training notebook: [Colab link](https://colab.research.google.com/drive/11Z1hxvwN6LVG1mC6Hx0y6iDFDRceNQR4?usp=sharing)
- MathBERT fine-tuning notebook: [Colab link](https://colab.research.google.com/drive/1SiXmUtuLf4ymV4njTwA__kA7EkJH1FHx?usp=sharing)

## Evaluation Metrics

- Per-label accuracy
- Jaccard score (samples average)
- F1 (micro & macro)
- Precision / Recall (micro)

## Results

Each model was evaluated on its own test split, all large enough to give reliable estimates.

| Metric | XGBoost | MathBERT | LSTM |
|---|---|---|---|
| **Accuracy** | | | |
| ├ Algebra | 0.921 | 0.927 | 0.922 |
| ├ Number Theory | 0.921 | 0.926 | 0.915 |
| ├ Geometry | 0.956 | 0.957 | 0.956 |
| └ Combinatorics | 0.928 | 0.936 | 0.926 |
| Jaccard (samples) | 0.847 | 0.875 | 0.861 |
| F1 micro | 0.873 | 0.884 | 0.872 |
| F1 macro | 0.860 | 0.873 | 0.858 |
| Precision (micro) | 0.901 | 0.902 | 0.888 |
| Recall (micro) | 0.848 | 0.867 | 0.857 |

**Takeaway:** All three models perform well and land in a similar range. In principle, MathBERT — a transformer pretrained specifically on mathematical text — should be able to get much closer to 100%, so the fact that it doesn't outperform XGBoost and LSTM by more suggests the ceiling here isn't really about model capacity, but about the data itself (e.g. label noise or inherent ambiguity between topics like algebra/number theory or geometry/combinatorics). This is backed up by manual inspection: several misclassified examples turned out to be mislabeled in the data itself. Given that, the far cheaper XGBoost is the more practical choice.


