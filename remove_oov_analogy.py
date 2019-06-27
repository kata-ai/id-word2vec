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

ex = Experiment(name='id-word2vec-remove-oov-analogy')

# Setup Mongo observer
mongo_url = os.getenv('SACRED_MONGO_URL')
db_name = os.getenv('SACRED_DB_NAME')
if mongo_url is not None and db_name is not None:
    ex.observers.append(MongoObserver.create(url=mongo_url, db_name=db_name))


@ex.config
def default():
    # path to analogy task file
    analogy_path = 'analogy.txt'
    # path to vocab file, one word in each line
    vocab_path = 'vocab.txt'
    # file encoding to use
    encoding = 'utf-8'
    # whether to lowercase words in analogies
    lower = True


@ex.capture
def read_vocab(vocab_path, _log, encoding='utf-8'):
    _log.info('Reading vocabulary from %s', vocab_path)
    with open(vocab_path, encoding=encoding) as f:
        return {line.strip() for line in f}


@ex.automain
def remove_oov(analogy_path, _log, encoding='utf-8', lower=True):
    """Remove questions with OOV words from an analogy task file."""
    vocab = read_vocab()
    _log.info('Processing analogies from %s', analogy_path)
    with open(analogy_path, encoding=encoding) as f:
        for line in f:
            if line.startswith(':'):
                # Found a section title
                print(line.rstrip())
            else:
                should_print = True

                for ws in line.split():
                    # Handle synonyms separated by slash (/)
                    for w in ws.split('/'):
                        if lower:
                            w = w.lower()
                        if w not in vocab:
                            should_print = False
                            break
                    if not should_print:
                        break

                if should_print:
                    print(line.rstrip())
