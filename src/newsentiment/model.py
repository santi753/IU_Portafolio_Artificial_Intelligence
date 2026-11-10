"""Sentiment model pipelines: TF-IDF features plus a linear classifier.

Every model is a two-step scikit-learn ``Pipeline``:

* ``tfidf``: a ``TfidfVectorizer`` that lowercases the text. Stop words are
  deliberately kept: scikit-learn's English list contains words such as
  *up*, *down*, *less*, *more*, *not* and *no*, which carry the sentiment of
  financial news ("profit fell", "sales did not increase").
* ``clf``: the classifier, either Multinomial Naive Bayes (baseline) or
  logistic regression (main model).

Because the vectorizer is part of the pipeline, it is refitted inside every
cross-validation fold, so the vocabulary and IDF weights never see the
held-out sentences.
"""

from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from newsentiment import config

# Names accepted by build_pipeline()
CLASSIFIERS = ("nb", "logreg")


def build_pipeline(classifier: str = "logreg", **params: Any) -> Pipeline:
    """Build an unfitted TF-IDF + classifier pipeline.

    Args:
        classifier: ``"nb"`` for Multinomial Naive Bayes or ``"logreg"`` for
            logistic regression.
        **params: Optional pipeline parameters in scikit-learn's
            ``step__param`` notation, passed to ``Pipeline.set_params``, e.g.
            ``clf__class_weight="balanced"`` or ``tfidf__ngram_range=(1, 2)``.

    Returns:
        The pipeline with the steps ``("tfidf", ...)`` and ``("clf", ...)``.

    Raises:
        ValueError: If ``classifier`` is not one of :data:`CLASSIFIERS`.
    """
    if classifier == "nb":
        clf = MultinomialNB()
    elif classifier == "logreg":
        # max_iter is raised from the default 100 so the solver converges on
        # the sparse, high-dimensional TF-IDF features.
        clf = LogisticRegression(max_iter=1000, random_state=config.RANDOM_SEED)
    else:
        raise ValueError(f"Unknown classifier {classifier!r}; expected one of {CLASSIFIERS}.")

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(lowercase=True, stop_words=None)),
            ("clf", clf),
        ]
    )
    if params:
        pipeline.set_params(**params)
    return pipeline


def _to_json(value: Any) -> Any:
    """Convert a parameter value into a JSON-serialisable equivalent."""
    if isinstance(value, tuple):
        return list(value)  # e.g. ngram_range (1, 2) -> [1, 2]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _to_json(v) for k, v in value.items()}
    return str(value)


def key_params(pipeline: Pipeline) -> dict:
    """Return the pipeline parameters worth logging for an iteration.

    Args:
        pipeline: A pipeline built by :func:`build_pipeline`.

    Returns:
        A JSON-serialisable dict with the vectorizer settings (``ngram_range``,
        ``min_df``, ``max_df``, ``sublinear_tf``), the classifier type and its
        main hyperparameters (``alpha`` and ``fit_prior`` for Naive Bayes; ``C`` and
        ``class_weight`` for logistic regression).
    """
    tfidf = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]

    params: dict[str, Any] = {
        "classifier": type(clf).__name__,
        "tfidf__ngram_range": tfidf.ngram_range,
        "tfidf__min_df": tfidf.min_df,
        "tfidf__max_df": tfidf.max_df,
        "tfidf__sublinear_tf": tfidf.sublinear_tf,
    }
    if isinstance(clf, MultinomialNB):
        params["clf__alpha"] = clf.alpha
        params["clf__fit_prior"] = clf.fit_prior
    elif isinstance(clf, LogisticRegression):
        params["clf__C"] = clf.C
        params["clf__class_weight"] = clf.class_weight
    return {name: _to_json(value) for name, value in params.items()}
