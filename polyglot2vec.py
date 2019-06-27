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

import os
import pickle

from sacred import Experiment
from sacred.observers import MongoObserver

ex = Experiment(name='id-word2vec-polyglot2vec')

# Setup Mongo observer
mongo_url = os.getenv('SACRED_MONGO_URL')
db_name = os.getenv('SACRED_DB_NAME')
if mongo_url is not None and db_name is not None:
    ex.observers.append(MongoObserver.create(url=mongo_url, db_name=db_name))


@ex.config
def default():
    # path to polyglot pretrained vectors
    path = ''
    # string encoding to use
    encoding = 'latin1'


@ex.automain
def convert(path, encoding='latin1'):
    """Convert Polyglot's word vectors into word2vec format."""
    with open(path, 'rb') as f:
        tokens, vectors = pickle.load(f, encoding=encoding)

    if len(tokens) != vectors.shape[0]:
        raise ValueError('length of vectors and tokens mismatch')

    print(vectors.shape[0], vectors.shape[1])
    for tok, vec in zip(tokens, vectors):
        print(tok, ' '.join(str(v) for v in vec))
