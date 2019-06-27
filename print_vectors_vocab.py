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

import gzip
import os

from sacred import Experiment
from sacred.observers import MongoObserver
from tqdm import tqdm

ex = Experiment(name='id-word2vec-print-vectors-vocab')

# Setup Mongo observer
mongo_url = os.getenv('SACRED_MONGO_URL')
db_name = os.getenv('SACRED_DB_NAME')
if mongo_url is not None and db_name is not None:
    ex.observers.append(MongoObserver.create(url=mongo_url, db_name=db_name))


@ex.config
def default():
    # path to vectors file in word2vec format (can be gzipped)
    path = 'vectors.txt'
    # file encoding to use
    encoding = 'utf-8'


@ex.automain
def print_vocab(path, encoding='utf-8'):
    """Print vocabulary of the given vectors file."""
    vocab = set()
    open_fn = gzip.open if path.endswith('.gz') else open

    with open_fn(path, 'rb') as f:
        header = next(f).decode(encoding)
        total = int(header.split()[0])
        for line in tqdm(f, total=total):
            line = line.decode(encoding)
            ents = line.split()
            k = 0
            while k < len(ents):
                try:
                    float(ents[k])
                except ValueError:
                    k += 1
                else:
                    break
            vocab.add(' '.join(ents[:k]))

    for w in sorted(vocab):
        print(w)
