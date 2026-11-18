# IU_Portafolio_Artificial_Intelligence

**Status:** Phase 2 submitted (18 Nov 2026); waiting for tutor feedback

A Python command-line tool that fetches recent Google News headlines for a company and
classifies each one as negative, neutral or positive. The classifier is a TF-IDF + logistic
regression model trained on the Financial PhraseBank; training and evaluation are documented
in Jupyter notebooks.

## Structure

```
src/newsentiment/    Python package
  news_fetcher.py    component 1: download Google News RSS headlines (no API key)
  preprocessing.py   component 2: strip publisher suffix, drop press releases, quote pages,
                     off-topic items and duplicates
  classifier.py      component 3: sentiment label + confidence per headline (saved model)
  reporter.py        component 4: summary statistics, console report, CSV export
  app.py             command-line application chaining the four components
  config.py, data.py, model.py, evaluation.py, validation.py, plotting.py   supporting modules
notebooks/           01 EDA, 02 training & evaluation, 03 real-news validation & demo
tests/               pytest unit tests (offline)
data/raw/            training data (not in the repository, see below)
data/validation/     30 hand-labelled real headlines, frozen with a SHA-256 hash
models/              final pipeline (sentiment_model.joblib) and its metadata
outputs/             metrics/ and figures/ from the notebooks; results/ from the CLI
```

## Setup (Windows, PowerShell, Python 3.12)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`requirements.txt` pins the tested versions and installs the package itself in editable mode.

## Training data

Only needed to re-run the notebooks; the trained model is included, so the CLI and the tests
work without it. Download "Sentiment Analysis for Financial News" from
[Kaggle](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news) and put
`all-data.csv` and the four `Sentences_*Agree.txt` files (folder `FinancialPhraseBank-v1.0` of the
download) into `data/raw/`. The data is the Financial PhraseBank of
Malo et al. (2014), licensed under CC BY-NC-SA 3.0; it is not redistributed in this repository.

## Usage

```powershell
python -m newsentiment <company> [--days N] [--top N] [--no-save]
python -m newsentiment Tesla --days 7 --top 10
```

`--days` sets the look-back window (default 7), `--top` the number of newest headlines listed
(default 10), and `--no-save` skips writing the raw news cache (`data/news/`) and the result CSVs
(`outputs/results/`). Fetched headlines and results are third-party content: both folders are
git-ignored and stay local. The only exception is the set of 30 hand-labelled validation
headlines used for the evaluation.

## Notebooks

Run in this order (needs the training data):

1. `01_eda.ipynb` — data cleaning and exploratory analysis
2. `02_training_evaluation.ipynb` — baseline, iterations, cross-validation, final model, test set
3. `03_validation_demo.ipynb` — evaluation on the real headlines and an application demo

## Tests

```powershell
pytest -v
```

The tests run offline. Tests that need the git-ignored files in `data/raw/` are skipped when
the files are missing.

## Key results and limitation

- Locked test set (n = 968): macro-F1 **0.740** (accuracy 0.783).
- Real Google News headlines (n = 30, labelled by hand): macro-F1 **0.348** (accuracy 0.467).
- Main limitation: a domain gap. The model learned from full sentences of financial news texts,
  while headlines are short and phrased differently; headline-style negative news is missed
  (none of the 6 negative headlines is recognised; all are predicted neutral).

## Troubleshooting

- `BrokenProcessPool` while running notebook 02 on Windows: the parallel grid search lost a
  worker process. Re-run the notebook.
- `FileNotFoundError` for `all-data.csv`: the training data is missing in `data/raw/` (see above).
