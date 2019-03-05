#!/bin/bash

OUTPUT_PATH=model/wikipedia-ja-d2v-model/doc2vec.gensim.model
DIC_PATH=/usr/local/lib/mecab/dic/mecab-ipadic-neologd
CORPUS_PATH=../../dataset/ArtistReviewCorpus_20180608
SIZE=200
WINDOW=8
MIN_COUNT=1
DM=0

# download mecab-ipadic-neologd
# python src/train_text_corpus.py --download-neologd --dictionary-path=$DIC_PATH

python src/train_text_corpus.py --build-model -o $OUTPUT_PATH --dictionary-path=$DIC_PATH --corpus-path=$CORPUS_PATH --size=$SIZE --window=$WINDOW --min-count=$MIN_COUNT --DM-$DM
