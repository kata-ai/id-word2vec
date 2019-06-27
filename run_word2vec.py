#!/usr/bin/env python

##########################################################################
# Copyright 2019 Kata.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##########################################################################

from itertools import chain
import os
import warnings

from gensim.models import FastText, Word2Vec
from gensim.models.word2vec import FAST_VERSION
from sacred import Experiment
from sacred.observers import MongoObserver

from ingredients.corpus import ing as corpus_ing, read_corpus
from ingredients.preprocess import ing as prep_ing, make_prep_sent

ex = Experiment(name='id-word2vec-default-word2vec', ingredients=[corpus_ing, prep_ing])

# Setup Mongo observer
mongo_url = os.getenv('SACRED_MONGO_URL')
db_name = os.getenv('SACRED_DB_NAME')
if mongo_url is not None and db_name is not None:
    ex.observers.append(MongoObserver.create(url=mongo_url, db_name=db_name))


@ex.config
def default():
    # dimension of the word embedding
    size = 300
    # context window size
    window = 15
    # discard words occurring fewer than this
    min_count = 5
    # number of epochs to train for
    epochs = 5
    # whether to use fastText instead
    use_fasttext = False
    # number of workers
    workers = os.cpu_count() - 1 if os.cpu_count() > 1 else 1
    # whether to save the vectors only
    vectors_only = True
    # where to save the result
    save_to = 'vectors.txt'


class SentencesCorpus:
    def __init__(self, read_corpus):
        self.read_corpus = read_corpus
        self.prep_sent = make_prep_sent()

    def __iter__(self):
        for paras in self.read_corpus():
            for sent in chain.from_iterable(paras):
                yield self.prep_sent(sent)


@ex.automain
def train(
        seed,
        _log,
        size=100,
        window=5,
        min_count=5,
        epochs=5,
        use_fasttext=False,
        workers=1,
        vectors_only=True,
        save_to='vectors.txt'):
    """Train word2vec/fastText word vectors."""
    if not FAST_VERSION:
        warnings.warn(
            "Gensim's FAST_VERSION is not set. Install C compiler before installing "
            "Gensim to get the fast version of word2vec.")

    cls = FastText if use_fasttext else Word2Vec

    _log.info('Start training')
    model = cls(
        SentencesCorpus(read_corpus),
        size=size,
        window=window,
        min_count=min_count,
        workers=workers,
        iter=epochs,
        seed=seed)

    _log.info('Training finished, saving model to %s', save_to)
    if vectors_only:
        model.wv.save_word2vec_format(save_to)
    else:
        model.save(save_to)
