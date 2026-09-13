#!/usr/bin/env bash
set -euo pipefail

# Run from the extracted delivery folder on Python 3.12.
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e .
.venv/bin/python -m pytest

# Optional executed-notebook verification.  The notebooks write only under
# their configured ignored output directory.
.venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=600 notebooks/Single_Allocation_Hub_Location_Complete.ipynb
.venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=600 notebooks/article_presentation_figures.ipynb
