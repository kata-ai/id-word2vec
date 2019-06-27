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

from sacred import Experiment
from tqdm import tqdm

from ingredients.corpus import ing as corpus_ing, read_corpus
from ingredients.preprocess import ing as prep_ing, make_prep_sent

ex = Experiment(ingredients=[corpus_ing, prep_ing])


@ex.automain
def print_stats():
    prep_sent = make_prep_sent()
    num_articles, num_tokens = 0, 0
    vocab = set()

    for paras in tqdm(read_corpus(), unit='doc'):
        num_articles += 1
        for sent in chain.from_iterable(paras):
            sent = prep_sent(sent)
            num_tokens += len(sent)
            vocab.update(sent)

    print('# articles    :', num_articles)
    print('# word tokens :', num_tokens)
    print('# word types  :', len(vocab))
