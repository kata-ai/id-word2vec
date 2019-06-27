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

from sacred import Experiment
from sacred.observers import MongoObserver

ex = Experiment(name='id-word2vec-make-shared-vocab')

# Setup Mongo observer
mongo_url = os.getenv('SACRED_MONGO_URL')
db_name = os.getenv('SACRED_DB_NAME')
if mongo_url is not None and db_name is not None:
    ex.observers.append(MongoObserver.create(url=mongo_url, db_name=db_name))


@ex.config
def default():
    # comma-separated paths of vocab files
    paths = 'vocab1.txt,vocab2.txt'
    # file encodings to use
    encodings = 'utf-8'


@ex.capture
def get_vocab(path, _log, encoding='utf-8'):
    with open(path, encoding=encoding) as f:
        return {line.strip() for line in f}


@ex.automain
def make_shared(paths, encodings):
    """Make a shared vocab from the vocab files."""
    shared_vocab = None
    for path in paths.split(','):
        vocab = get_vocab(path)
        if shared_vocab is None:
            shared_vocab = vocab
        else:
            shared_vocab.intersection_update(vocab)

    for w in sorted(shared_vocab):
        print(w)
