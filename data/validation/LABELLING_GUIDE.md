# Labelling guide: real-news validation set

30 real headlines (10 each for Tesla, Airbus and Siemens) are labelled by hand
to validate the sentiment model. The labels are frozen afterwards and never used
to tune the model.

## The question

For each headline, ask:

> Would an investor read this headline as good news, bad news, or neither for
> the company?

Write exactly one label in the `label` column: `positive`, `negative` or `neutral`.

| Label | Typical headlines |
|---|---|
| `positive` | rising sales or profit, big orders, records, successful launches, analyst upgrades, legal wins |
| `negative` | falling sales, losses, recalls, cancellations, lawsuits, accidents, layoffs, downgrades, share price falls |
| `neutral` | factual announcements without a clear direction, appointments, previews, questions, headlines that mix good and bad |

## Rules

- Judge the headline only. Do not open or search for the article.
- Label before the model is run on these headlines, and do not look at earlier
  reports or predictions for these companies while labelling.
- Never change a label after the evaluation. Once frozen, the file's SHA-256 is
  checked, so any edit is detected.

## Steps

1. Open `real_news_to_label.csv` in Excel. If everything appears in one column,
   use *Data → From Text/CSV* and choose the comma as delimiter.
2. Fill in the `label` column for all 30 rows. Do not change the other columns.
3. Save as `real_news_labelled.csv` in this folder, preferably as *CSV UTF-8*.
   Semicolon-separated files (e.g. Excel with French regional settings) are fine.
4. Freeze the labels: `python -m newsentiment.validation freeze`
