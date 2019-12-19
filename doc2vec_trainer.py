import functools
from pathlib import Path
import multiprocessing
import logging
import more_itertools
from gensim.models.doc2vec import TaggedDocument, Doc2Vec

logging.basicConfig(level=logging.INFO)


def train_doc2vec_model(output_model_path, iter_docs, tokenizer, size=400, window=8, min_count=5, dm=1, epoch=5):
    """
    Parameters
    ----------
    output_model_path : string
        path of Doc2Vec model
    iter_docs : iterator
        iterator of documents, which are raw texts
    tokenizer : subclass of DocumentTokenizerBase
        word tokenizer
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
    logging.info("get tokens iteractor")

    iter_tokens = tokenizer.get_tokens_iterator(iter_docs, normalize=False)

    logging.info("count number of documents")
    
    n_obs = more_itertools.ilen(iter_tokens())

    logging.info("build vocabulary")

    model = Doc2Vec(
        vector_size=size,
        window=window,
        min_count=min_count,
        dm=dm,
        workers=multiprocessing.cpu_count()
    )
    model.build_vocab(iter_tokens())

    logging.info("train doc2vec")

    model.train(iter_tokens(), total_examples=n_obs, epochs=epoch)
    model.init_sims(replace=True)

    logging.info("save model")

    p = Path(output_model_path)
    if not p.parent.exists():
        p.parent.mkdir(parents=True)
    model.save(output_model_path)

    logging.info("done.")
