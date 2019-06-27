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
from pathlib import Path
import gzip
import json

from sacred import Ingredient

ing = Ingredient('corpus')


@ing.config
def default():
    # path to the corpus directory
    path = 'tempo'
    # product type (kt, mbm, all)
    product = 'all'
    # read kt corpus starting from this year
    kt_begin = 2005
    # stop reading kt corpus at this year (inclusive)
    kt_end = 2014
    # read mbm corpus starting from this year
    mbm_begin = 1999
    # stop reading mbm corpus at this year (inclusive)
    mbm_end = 2014
    # file encoding to use
    encoding = 'utf-8'


@ing.capture
def read_corpus(
        path,
        _log,
        product='all',
        kt_begin=2005,
        kt_end=2014,
        mbm_begin=1999,
        mbm_end=2014,
        encoding='utf-8'):
    path = Path(path)

    if product in ('kt', 'mbm'):
        path = path / product
        begin, end = (kt_begin, kt_end) if product == 'kt' else (mbm_begin, mbm_end)
        _log.info('Reading corpus from %s year %s-%s', path, begin, end)
        return _read(path, begin, end, encoding=encoding)

    assert product == 'all', "product must be one of 'kt', 'mbm', or 'all'"

    _log.info('Reading corpus from %s year %s-%s', path / 'kt', kt_begin, kt_end)
    kt_corpus = _read(path / 'kt', kt_begin, kt_end, encoding=encoding)

    _log.info('Reading corpus from %s year %s-%s', path / 'mbm', mbm_begin, mbm_end)
    mbm_corpus = _read(path / 'mbm', mbm_begin, mbm_end, encoding=encoding)

    return chain(kt_corpus, mbm_corpus)


def _read(corpus_dir, begin_year, end_year, encoding='utf-8'):
    for year in range(begin_year, end_year + 1):
        path = corpus_dir / f'{year}.jsonl'
        if not path.exists():
            path = corpus_dir / f'{year}.jsonl.gz'
        open_fn = gzip.open if path.name.endswith('.gz') else open

        with open_fn(path, 'rb') as f:
            for line in f:
                yield json.loads(line.decode(encoding).strip())['paragraphs']
