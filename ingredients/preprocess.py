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

import re

from sacred import Ingredient

ing = Ingredient('prep')


@ing.config
def default():
    # whether to lowercase words
    lower = True
    # remove words not matching this pattern
    word_pattern = r'[\w\-]+$'
    # whether to map numbers to a single token
    map_numbers = True
    # the special token to map numbers to
    number_token = '@@NUM@@'


@ing.capture
def make_prep_sent(
        lower=True, word_pattern=r'[\w\-]+$', map_numbers=True, number_token='@@NUM@@'):
    word_re = re.compile(word_pattern)
    number_re = re.compile(r'\d+$')

    def prep_sent(sent):
        if lower:
            sent = [w.lower() for w in sent]
        sent = [w for w in sent if word_re.match(w)]
        if map_numbers:
            sent = [number_token if number_re.match(w) else w for w in sent]
        return sent

    return prep_sent
