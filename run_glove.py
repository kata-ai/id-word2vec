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

from pathlib import Path
import os
import sys

from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds

ex = Experiment(name='id-word2vec-default-glove')
ex.captured_out_filter = apply_backspaces_and_linefeeds

# Setup Mongo observer
mongo_url = os.getenv('SACRED_MONGO_URL')
db_name = os.getenv('SACRED_DB_NAME')
if mongo_url is not None and db_name is not None:
    ex.observers.append(MongoObserver.create(url=mongo_url, db_name=db_name))


@ex.config
def default():
    # path to the corpus file
    corpus = 'corpus.txt'
    # dimension of the word embedding
    size = 100
    # context window size
    window = 10
    # discard words occurring fewer than this
    min_count = 5
    # number of epochs to train for
    epochs = 50
    # number of workers
    workers = os.cpu_count() - 1 if os.cpu_count() > 1 else 1
    # output (and working) directory
    outdir = 'output'
    # glove binary directory (empty string == binaries are in PATH)
    bindir = ''


VOCAB_FNAME = 'vocab.txt'
COOCCUR_FNAME = 'cooccurences.bin'
SHUF_FNAME = 'cooccurences.shuf.bin'
VECTORS_FNAME = 'vectors'


def runcmd(cmd):
    rc = os.system(cmd)
    if rc != 0:
        sys.exit(rc)


@ex.command
def vocab_count(corpus, min_count=5, outdir='output', bindir=''):
    """Run GloVe's vocab_count."""
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True)
    vocab = outdir / VOCAB_FNAME

    cmd = os.path.join(bindir, 'vocab_count')
    cmd += f' -min-count {min_count}'
    cmd += f' < {corpus} > {vocab}'

    runcmd(cmd)


@ex.command
def cooccur(corpus, window=10, outdir='output', bindir=''):
    """Run GloVe's cooccur."""
    outdir = Path(outdir)
    vocab = outdir / VOCAB_FNAME
    cooccur_path = outdir / COOCCUR_FNAME
    overflow = outdir / 'overflow'

    cmd = os.path.join(bindir, 'cooccur')
    cmd += f' -vocab-file {vocab}'
    cmd += f' -window-size {window}'
    cmd += f' -overflow-file {overflow}'
    cmd += f' < {corpus} > {cooccur_path}'

    runcmd(cmd)


@ex.command
def shuffle(outdir='output', bindir=''):
    """Run GloVe's shuffle."""
    outdir = Path(outdir)
    cooccur_path = outdir / COOCCUR_FNAME
    shuf = outdir / SHUF_FNAME
    temp = outdir / 'temp_shuffle'

    cmd = os.path.join(bindir, 'shuffle')
    # verbose flag needed, see https://github.com/stanfordnlp/GloVe/issues/137
    cmd += f' -verbose 2'
    cmd += f' -temp-file {temp}'
    cmd += f' < {cooccur_path} > {shuf}'

    runcmd(cmd)


@ex.command
def glove(size=100, workers=1, epochs=50, outdir='output', bindir=''):
    """Run GloVe's glove."""
    outdir = Path(outdir)
    vocab = outdir / VOCAB_FNAME
    shuf = outdir / SHUF_FNAME
    vectors = outdir / VECTORS_FNAME

    cmd = os.path.join(bindir, 'glove')
    cmd += f' -vector-size {size}'
    cmd += f' -threads {workers}'
    cmd += f' -iter {epochs}'
    cmd += f' -input-file {shuf}'
    cmd += f' -vocab-file {vocab}'
    cmd += f' -save-file {vectors}'
    cmd += f' -binary 0'  # save as text
    cmd += f' -model 1'  # save only the word vectors

    runcmd(cmd)


@ex.automain
def train():
    """Train GloVe word vectors from scratch."""
    vocab_count()
    cooccur()
    shuffle()
    glove()
