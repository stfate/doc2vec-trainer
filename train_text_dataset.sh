#!/bin/bash

DATASET_PATH=../../dataset/ArtistReviewCorpus_20180608
LANG=ja
DIC_PATH=/usr/local/lib/mecab/dic/mecab-ipadic-neologd
OUTPUT_PATH=model/wikipedia-ja-d2v-model/doc2vec.gensim.model
SIZE=400
WINDOW=8
MIN_COUNT=1
DM=0

# download mecab-ipadic-neologd
# python src/train_text_dataset.py --download-neologd --dictionary-path=$DIC_PATH

python src/train_text_dataset.py --build-model -o $OUTPUT_PATH --dictionary-path=$DIC_PATH --dataset-path=$DATASET_PATH --lang=$LANG --size=$SIZE --window=$WINDOW --min-count=$MIN_COUNT --DM-$DM