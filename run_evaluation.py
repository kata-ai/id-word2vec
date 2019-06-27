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

from collections import defaultdict
from typing import Tuple
import os
import random

from gensim.models import KeyedVectors
from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds
from tqdm import trange
import numpy as np

ex = Experiment(name='id-word2vec-eval-ci')
ex.captured_out_filter = apply_backspaces_and_linefeeds

# Setup Mongo observer
mongo_url = os.getenv('SACRED_MONGO_URL')
db_name = os.getenv('SACRED_DB_NAME')
if mongo_url is not None and db_name is not None:
    ex.observers.append(MongoObserver.create(url=mongo_url, db_name=db_name))


@ex.config
def default():
    # path to the word vectors file
    vectors_path = 'vectors.txt'
    # path to the analogy task file in Google's format
    analogy_path = 'analogy.txt'
    # file encoding to use
    encoding = 'utf-8'
    # whether to lowercase the analogies
    lower = True
    # compute acccuracy at this number of most similar results
    at = 1
    # whether to skip analogy containing OOV words
    skip_oov = True
    # significance level
    alpha = 0.95
    # number of bootstrap samples
    n_samples = 1000


@ex.capture
def load_word_vectors(
        _log,
        vectors_path: str = 'vectors.txt',
        encoding: str = 'utf-8',
) -> KeyedVectors:
    _log.info('Loading word vectors from %s', vectors_path)
    return KeyedVectors.load_word2vec_format(vectors_path, encoding=encoding)


Analogy = Tuple[str, str, str, str]


@ex.capture
def is_correct(kv: KeyedVectors, analogy: Analogy, at: int = 1) -> bool:
    pos = analogy[0].split('/') + analogy[2].split('/')
    neg = analogy[1].split('/')
    tgt = analogy[3].split('/')

    sim_words = set(w for w, _ in kv.most_similar(positive=pos, negative=neg, topn=at))
    return any(t in sim_words for t in tgt)


@ex.capture
def compute_bootstrap_ci(samples, _log, alpha=0.95, n_samples=1000):
    # See https://www2.stat.duke.edu/~banks/111-lectures.dir/lect13.pdf
    _log.info('Computing confidence interval via bootstrapping')
    bs_means = []
    for _ in trange(n_samples):
        bs_means.append(np.mean(random.choices(samples, k=len(samples))))

    qlo = 0.5 * (1 - alpha)
    qhi = 1 - qlo
    bs_mean_lo, bs_mean_hi = np.quantile(bs_means, [qlo, qhi])
    mean = np.mean(samples)
    return (2 * mean - bs_mean_hi, 2 * mean - bs_mean_lo)


@ex.capture
def get_corrects(kv, stream, _log, lower=True, skip_oov=True):
    corrects = defaultdict(list)
    section = ''

    for linum, line in enumerate(stream, 1):
        if line.startswith(': '):
            section = line[2:].strip()
        else:
            if lower:
                line = line.lower()
            analogy = tuple(line.split())
            if len(analogy) != 4:
                raise ValueError(
                    f'analogy at line {linum} has {len(analogy)} entries, expected 4')

            try:
                correct = is_correct(kv, analogy)
            except KeyError as e:
                if skip_oov:
                    _log.debug('%s, skipping', e)
                    continue
                else:
                    _log.debug('%s, assuming incorrect', e)
                    correct = False
            corrects[section].append(1 if correct else 0)

    return corrects


@ex.command
def print_corrects(_log, analogy_path: str = 'analogy.txt'):
    """Print 0/1 labels indicating if the analogy is correct/not."""
    kv = load_word_vectors()
    _log.info('Reading analogies from %s', analogy_path)
    with open(analogy_path) as f:
        corrects = get_corrects(kv, f)
    for sec in sorted(corrects):
        cs = corrects[sec]
        print('\n'.join(str(c) for c in cs))


@ex.automain
def evaluate(_log, _run, analogy_path: str = 'analogy.txt'):
    """Evaluate a given word vectors on word analogy task."""
    kv = load_word_vectors()
    _log.info('Reading analogies from %s', analogy_path)
    with open(analogy_path) as f:
        corrects = get_corrects(kv, f)
    _log.info('Accuracies:')
    for sec, cs in corrects.items():
        acc = np.mean(cs)
        _run.log_scalar(f'acc({sec})', acc)
        _log.info(f'{sec} : {acc:.2%}')

    all_corrects = [c for cs in corrects.values() for c in cs]
    acc = np.mean(all_corrects)
    _run.log_scalar('acc(**overall**)', acc)
    _log.info(f'**overall** : {acc:.2%}')

    acc_lo, acc_hi = compute_bootstrap_ci(all_corrects)
    _run.log_scalar('acc_lo(**overall**)', acc_lo)
    _run.log_scalar('acc_hi(**overall**)', acc_hi)
    _log.info(f'Confidence interval: [{acc_lo:.2%}, {acc_hi:.2%}]')

    return acc
