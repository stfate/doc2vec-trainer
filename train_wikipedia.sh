#!/bin/bash

WIKIPEDIA_DUMP_PATH=../../dataset/Wikipedia/jawiki-latest-pages-articles.xml.bz2
LANG=ja
DIC_PATH=/usr/local/lib/mecab/dic/mecab-ipadic-neologd
OUTPUT_PATH=model/wikipedia-ja-d2v-model/doc2vec.gensim.model
SIZE=400
WINDOW=8
MIN_COUNT=5
DM=0

# download wikipedia dump
# python src/train_wikipedia.py --download-wikipedia-dump --wikipedia-dump-path=$WIKIPEDIA_DUMP_PATH

# download mecab-ipadic-neologd
# python src/train_wikipedia.py --download-neologd --dictionary-path=$DIC_PATH

# train doc2vec model
python src/train_wikipedia.py --build-model -o $OUTPUT_PATH --dictionary-path=$DIC_PATH --wikipedia-dump-path=$WIKIPEDIA_DUMP_PATH --lang=$LANG --size=$SIZE --window=$WINDOW --min-count=$MIN_COUNT --dm=$DM
