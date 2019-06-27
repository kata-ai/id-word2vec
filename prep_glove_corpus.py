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

from sacred import Experiment
from sacred.observers import MongoObserver
from tqdm import tqdm

from ingredients.corpus import ing as corpus_ing, read_corpus
from ingredients.preprocess import ing as prep_ing, make_prep_sent

ex = Experiment(name='id-word2vec-prep-glove-corpus', ingredients=[corpus_ing, prep_ing])

# Setup Mongo observer
mongo_url = os.getenv('SACRED_MONGO_URL')
db_name = os.getenv('SACRED_DB_NAME')
if mongo_url is not None and db_name is not None:
    ex.observers.append(MongoObserver.create(url=mongo_url, db_name=db_name))


@ex.automain
def prepare():
    """Prepare corpus for training with GloVe."""
    prep_sent = make_prep_sent()
    for paras in tqdm(read_corpus(), unit='doc'):
        sents = []
        for sent in chain.from_iterable(paras):
            sents.append(' '.join(prep_sent(sent)))
        print(' '.join(sents))
