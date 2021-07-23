# MIT License

# Copyright (c) 2021 FederalLab

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import collections
import json
import os
import re
from typing import Dict
import h5py
import numpy as np
from openfed.common import logger
from typing import Callable
from ..datasets import FederatedDataset
from ..utils import *

DEFAULT_BATCH_SIZE      = 4
DEFAULT_TRAIN_FILE_PROX = 'all_data_niid_2_keep_0_train_8.json'
DEFAULT_TEST_FILE_PROX  = 'all_data_niid_2_keep_0_test_8.json'

# group name defined by tff in h5 file
_USERS         = 'users'
_SNIPPETS_PROX = 'user_data'

# ------------------------
# utils for shakespeare dataset

# Vocabulary re-used from the Federated Learning for Text Generation tutorial.
# https://www.tensorflow.org/federated/tutorials/federated_learning_for_text_generation
CHAR_VOCAB = list(
    'dhlptx@DHLPTX $(,048cgkoswCGKOSW[_#\'/37;?bfjnrvzBFJNRVZ"&*.26:\naeimquyAEIMQUY]!%)-159\r'
)
ALL_LETTERS = "".join(CHAR_VOCAB)

# Vocabulary with OOV ID, zero for the padding, and BOS, EOS IDs.
VOCAB_SIZE = len(ALL_LETTERS) + 4


def _one_hot(index, size):
    '''returns one-hot vector with given size and value 1 at given index
    '''
    vec = [0 for _ in range(size)]
    vec[int(index)] = 1
    return vec


def letter_to_vec(letter):
    '''returns one-hot representation of given letter
    '''
    index = ALL_LETTERS.find(letter)
    return _one_hot(index, VOCAB_SIZE)


def letter_to_index(letter):
    '''returns one-hot representation of given letter
    '''
    index = ALL_LETTERS.find(letter)
    return index


def word_to_indices(word):
    '''returns a list of character indices

    Args:
        word: string

    Return:
        indices: int list with length len(word)
    '''
    indices = []
    for c in word:
        indices.append(ALL_LETTERS.find(c))
    return indices


# ------------------------
# utils for sent140 dataset


def split_line(line):
    '''split given line/phrase into list of words

    Args:
        line: string representing phrase to be split

    Return:
        list of strings, with each string representing a word
    '''
    return re.findall(r"[\w']+|[.,!?;]", line)


def _word_to_index(word, indd):
    '''returns index of given word based on given lookup dictionary

    returns the length of the lookup dictionary if word not found

    Args:
        word: string
        indd: dictionary with string words as keys and int indices as values
    '''
    if word in indd:
        return indd[word]
    else:
        return len(indd)


def line_to_indices(line, word2id, max_words=25):
    '''converts given phrase into list of word indices

    if the phrase has more than max_words words, returns a list containing
    indices of the first max_words words
    if the phrase has less than max_words words, repeatedly appends integer 
    representing unknown index to returned list until the list's length is 
    max_words

    Args:
        line: string representing phrase/sequence of words
        word2id: dictionary with string words as keys and int indices as values
        max_words: maximum number of word indices in returned list

    Return:
        indl: list of word indices, one index for each word in phrase
    '''
    unk_id    = len(word2id)
    line_list = split_line(line)  # split phrase in words
    indl  = [word2id[w] if w in word2id else unk_id for w in line_list[:max_words]]
    indl += [unk_id]*(max_words-len(indl))
    return indl


def bag_of_words(line, vocab):
    '''returns bag of words representation of given phrase using given vocab

    Args:
        line: string representing phrase to be parsed
        vocab: dictionary with words as keys and indices as values

    Return:
        integer list
    '''
    bag   = [0]*len(vocab)
    words = split_line(line)
    for w in words:
        if w in vocab:
            bag[vocab[w]] += 1
    return bag


DEFAULT_TRAIN_CLIENTS_NUM = 715
DEFAULT_TEST_CLIENTS_NUM  = 715
DEFAULT_BATCH_SIZE        = 4
DEFAULT_TRAIN_FILE_TFF    = 'shakespeare_train.h5'
DEFAULT_TEST_FILE_TFF     = 'shakespeare_test.h5'

# group name defined by tff in h5 file
_EXAMPLE      = 'examples'
_SNIPPETS_TFF = 'snippets'


word_dict = None
word_list = None
_pad      = '<pad>'
_bos      = '<bos>'
_eos      = '<eos>'
'''
This code follows the steps of preprocessing in tff shakespeare dataset: 
https://github.com/google-research/federated/blob/master/utils/datasets/shakespeare_dataset.py
'''

SEQUENCE_LENGTH = 80  # from McMahan et al AISTATS 2017


def get_word_dict():
    global word_dict
    if word_dict == None:
        words     = [_pad] + CHAR_VOCAB + [_bos] + [_eos]
        word_dict = collections.OrderedDict()
        for i, w in enumerate(words):
            word_dict[w] = i
    return word_dict


def get_word_list():
    global word_list
    if word_list == None:
        word_dict = get_word_dict()
        word_list = list(word_dict.keys())
    return word_list


def id_to_word(idx):
    return get_word_list()[idx]


def char_to_id(char):
    word_dict = get_word_dict()
    if char in word_dict:
        return word_dict[char]
    else:
        return len(word_dict)


def preprocess(sentences, max_seq_len=SEQUENCE_LENGTH):

    sequences = []

    def to_ids(sentence, num_oov_buckets=1):
        '''
        map list of sentence to list of [idx..] and pad to max_seq_len + 1
        Args:
            num_oov_buckets : The number of out of vocabulary buckets.
            max_seq_len: Integer determining shape of padded batches.
        '''
        tokens = [char_to_id(c) for c in sentence]
        tokens = [char_to_id(_bos)] + tokens + [char_to_id(_eos)]
        if len(tokens) % (max_seq_len + 1) != 0:
            pad_length  = (-len(tokens)) % (max_seq_len + 1)
            tokens     += [char_to_id(_pad)] * pad_length
        return (tokens[i:i + max_seq_len + 1]
                for i in range(0, len(tokens), max_seq_len + 1))

    for sen in sentences:
        sequences.extend(to_ids(sen))
    return sequences


class ShakespeareNWP(FederatedDataset):
    """Federated Shakespeare Dataset from [TFF](https://github.com/tensorflow/federated).
    Used for next word prediction.

    Downloads and caches the dataset locally. If previously downloaded, tries to
    load the dataset from cache.
    This dataset is derived from the Leaf repository
    (https://github.com/TalwalkarLab/leaf) pre-processing on the works of
    Shakespeare, which is published in "LEAF: A Benchmark for Federated Settings"
    https://arxiv.org/abs/1812.01097.
    The data set consists of 715 users (characters of Shakespeare plays), where
    each
    example corresponds to a contiguous set of lines spoken by the character in a
    given play.
    Data set sizes:
    -   train: 16,068 examples
    -   test: 2,356 examples
    Rather than holding out specific users, each user's examples are split across
    _train_ and _test_ so that all users have at least one example in _train_ and
    one example in _test_. Characters that had less than 2 examples are excluded
    from the data set.
    The `tf.data.Datasets` returned by
    `tff.simulation.datasets.ClientData.create_tf_dataset_for_client` will yield
    `collections.OrderedDict` objects at each iteration, with the following keys
    and values:
        -   `'snippets'`: a `tf.Tensor` with `dtype=tf.string`, the snippet of
        contiguous text.
    """

    def __init__(self, 
                root    : str,
                train   : bool = True,
                download: bool = True,
                transform: Callable=None): 
        data_file = os.path.join(
            root, DEFAULT_TRAIN_FILE_TFF if train else DEFAULT_TEST_FILE_TFF)

        if not os.path.isfile(data_file):
            if download:
                url = 'https://fedml.s3-us-west-1.amazonaws.com/shakespeare.tar.bz2'
                logger.debug(f"Download dataset from {url} to {root}")
                if wget_https(url, root):
                    if tar_xvf(os.path.join(root, "shakespeare.tar.bz2"), output_dir=root):
                        logger.debug("Downloaded.")
                else:
                    raise RuntimeError("Download dataset failed.")
            else:
                raise FileNotFoundError(f"{data_file} not exists.")

        data_h5:Dict[str, Dict] = h5py.File(data_file, "r") # type: ignore
        client_ids       = list(data_h5[_EXAMPLE].keys())
        self.total_parts = len(client_ids)

        self.part_id    = 0
        parts_data_list = []
        for client_id in client_ids:
            parts_data_list.append(
                [x.decode('utf-8') for x in np.array(data_h5[_EXAMPLE][client_id][_SNIPPETS_TFF][()])])
        self.parts_data_list = parts_data_list

        self.classes   = len(get_word_dict()) + 1
        self.transform = transform

    def __len__(self) -> int:
        return len(self.parts_data_list[self.part_id])

    def __getitem__(self, index: int):
        data = preprocess([self.parts_data_list[self.part_id][index]])[
            0].reshape(-1)
        x, y = data[:-1], data[-1]
        if self.transform:
            x, y = self.transform(x), self.transform(y)
        return x, y

    def total_samples(self):
        return sum([len(x) for x in self.parts_data_list])


class ShakespeareNCP(FederatedDataset):
    """Used for next char prediction. FedProx version.
    """

    def __init__(self, 
                root     : str,
                train    : bool    = True,
                download : bool    = True,
                transform: Callable = None): 
        data_file = os.path.join(
            root, DEFAULT_TRAIN_FILE_PROX if train else DEFAULT_TEST_FILE_PROX)

        if not os.path.isfile(data_file):
            if download:
                file_ids = ['1mD6_4ju7n2WFAahMKDtozaGxUASaHAPH',
                            '1GERQ9qEJjXk_0FXnw1JbjuGCI-zmmfsk']
                filenames = [
                    DEFAULT_TRAIN_FILE_PROX,
                    DEFAULT_TEST_FILE_PROX,
                ]
                for file_id, filename in zip(file_ids, filenames):
                    file_id  = '1GERQ9qEJjXk_0FXnw1JbjuGCI-zmmfsk'
                    filename = os.path.join(root, filename)
                    logger.debug(f"Download dataset: {file_id}, {filename}")
                    if not wget_google_driver_url(file_id, filename):
                        raise RuntimeError("Download dataset failed.")
            else:
                raise FileNotFoundError(f"{data_file} not exists.")

        with open(data_file, "r") as f:
            data_json       = json.load(f)
            parts_data_list = data_json[_SNIPPETS_PROX]

        self.part_id     = 0
        self.total_parts = len(parts_data_list.keys())
        self.parts_name  = list(parts_data_list.keys())

        self.parts_data_list = parts_data_list

        self.classes   = VOCAB_SIZE
        self.transform = transform

    def __len__(self) -> int:
        part_name = self.parts_name[self.part_id]
        return len(self.parts_data_list[part_name]['x'])

    def __getitem__(self, index: int):
        part_name = self.parts_name[self.part_id]
        data      = self.parts_data_list[part_name]
        x, y = word_to_indices(
            data['x'][index]), letter_to_index(data['y'][index])
        if self.transform:
            x, y = self.transform(x), self.transform(y)
        return x, y

    def total_samples(self):
        return sum([len(x['x']) for x in self.parts_data_list.values()])
