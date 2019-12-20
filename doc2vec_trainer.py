import functools
from pathlib import Path
import multiprocessing
import logging
import more_itertools
from gensim.models.doc2vec import Doc2Vec

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s:%(message)s", datefmt="%Y/%m/%d %H:%M:%S")


def train_doc2vec_model(output_model_path, iter_docs, size=400, window=8, min_count=5, dm=1, epoch=5):
    """
    Parameters
    ----------
    output_model_path : string
        path of Doc2Vec model
    iter_docs : iterator
        iterator of documents, which are raw texts
    size : int
        size of paragraph vector
    window : int
        window size of doc2vec
    min_count : int
        minimum word count
    dm : int
        doc2vec training algorithm (1: distributed memory other: distributed bag of words)
    epoch : int
        number of epochs
    """
    logging.info("build vocabularies")

    model = Doc2Vec(
        vector_size=size,
        window=window,
        min_count=min_count,
        dm=dm,
        workers=multiprocessing.cpu_count()
    )
    model.build_vocab(iter_docs())

    logging.info("train doc2vec")

    model.train(iter_docs(), total_examples=model.corpus_count, epochs=epoch)
    model.init_sims(replace=True)

    logging.info("save model")

    p = Path(output_model_path)
    if not p.parent.exists():
        p.parent.mkdir(parents=True)
    model.save(output_model_path)

    logging.info("done.")
