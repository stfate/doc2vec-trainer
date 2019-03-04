#!/bin/bash

OUTPUT_PATH=model/wikipedia-ja-d2v-model/doc2vec.gensim.model
DIC_PATH=/usr/local/lib/mecab/dic/mecab-ipadic-neologd
WIKIPEDIA_DUMP_PATH=data/jawiki-latest-pages-articles.xml.bz2
SIZE=400
WINDOW=8
MIN_COUNT=5
DM=0

# download wikipedia dump
# python src/train_wikipedia.py --download-wikipedia-dump --wikipedia-dump-path=$WIKIPEDIA_DUMP_PATH

# download mecab-ipadic-neologd
# python src/train_wikipedia.py --download-neologd --dictionary-path=$DIC_PATH

python src/train_wikipedia.py --build-model -o $OUTPUT_PATH --dictionary-path=$DIC_PATH --wikipedia-dump-path=$WIKIPEDIA_DUMP_PATH --size=$SIZE --window=$WINDOW --min-count=$MIN_COUNT --dm=$DM
