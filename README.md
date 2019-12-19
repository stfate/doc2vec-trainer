doc2vec-trainer
===============================

# Overview

gensim doc2vecモデルの学習を行うツールキット．

# Requirements

- cURL
- MeCab == 0.996
- Python >= 3.6

# Setup

```bash
git submodule init
git submodule update
pip install -r requirements.txt
```

# Run

詳細は`train_wikipedia.sh`を参照されたい．

## General dataset

```bash
python train_text_dataset.py -o $OUTPUT_PATH --dictionary-path=$DIC_PATH --corpus-path=$CORPUS_PATH --size=100 --window=8 --min-count=5 --dm=0
```

# How to use model

```python
model_path = "model/doc2vec.gensim.model"
model = Doc2Vec.load(model_path)
```
