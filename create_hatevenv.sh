#!/usr/bin/env bash

VENVNAME=hate_venv

python3 -m venv $VENVNAME
source $VENVNAME/bin/activate
pip install --upgrade pip

# problems when installing from requirements.txt
brew install wget
pip install ipython
pip install jupyter
pip install transformers
pip install torch

python -m ipykernel install --user --name=$VENVNAME

test -f requirements.txt && pip install -r requirements.txt

echo "build $VENVNAME"