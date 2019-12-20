doc2vec-trainer
===============================

# Overview

gensim doc2vecモデルの学習を行うツールキット．


# Requirements

doc2vecはgensimの実装を使用．  
他の依存パッケージは`requirements.txt`を参照．


# Setup

```bash
git submodule init
git submodule update
pip install -r requirements.txt
```

# Run

## General dataset

```bash
python train_text_dataset.py -o $OUTPUT_PATH --dictionary-path=$DIC_PATH --corpus-path=$CORPUS_PATH --size=100 --window=8 --min-count=5 --dm=0
```


# How to use model

```python
model_path = "model/doc2vec.gensim.model"
model = Doc2Vec.load(model_path)
```
