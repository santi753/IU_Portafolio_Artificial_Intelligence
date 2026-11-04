# IU_Portafolio_Artificial_Intelligence

A Python command-line tool that fetches recent Google News headlines for a company and classifies each one as negative, neutral or positive. The classifier is a TF-IDF + logistic regression model trained on the Financial PhraseBank in Jupyter notebooks.

**Status:** Phase 2 – in development

## Setup (PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest
```
