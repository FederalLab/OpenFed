import collections
import json
import os
import re

import h5py
import numpy as np
import torch

from ..dataset import FederatedDataset

DEFAULT_BATCH_SIZE = 4
DEFAULT_TRAIN_FILE = 'all_data_niid_2_keep_0_train_8.json'
DEFAULT_TEST_FILE = 'all_data_niid_2_keep_0_test_8.json'

# group name defined by tff in h5 file
_USERS = 'users'
_SNIPPETS = 'user_data'

# ------------------------
# utils for shakespeare dataset

# Vocabulary re-used from the Federated Learning for Text Generation tutorial.
# https://www.tensorflow.org/federated/tutorials/federated_learning_for_text_generation
CHAR_VOCAB = list(
    'dhlptx@DHLPTX $(,048cgkoswCGKOSW[_#\'/37;?bfjnrvzBFJNRVZ"&*.26:\naeimquyAEIMQUY]!%)-159\r'
)

# ALL_LETTERS = "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
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
    unk_id = len(word2id)
    line_list = split_line(line)  # split phrase in words
    indl = [word2id[w] if w in word2id else unk_id for w in line_list[:max_words]]
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
    bag = [0]*len(vocab)
    words = split_line(line)
    for w in words:
        if w in vocab:
            bag[vocab[w]] += 1
    return bag


DEFAULT_TRAIN_CLIENTS_NUM = 715
DEFAULT_TEST_CLIENTS_NUM = 715
DEFAULT_BATCH_SIZE = 4
DEFAULT_TRAIN_FILE = 'shakespeare_train.h5'
DEFAULT_TEST_FILE = 'shakespeare_test.h5'

# group name defined by tff in h5 file
_EXAMPLE = 'examples'
_SNIPPETS = 'snippets'


word_dict = None
word_list = None
_pad = '<pad>'
_bos = '<bos>'
_eos = '<eos>'
'''
This code follows the steps of preprocessing in tff shakespeare dataset: 
https://github.com/google-research/federated/blob/master/utils/datasets/shakespeare_dataset.py
'''

SEQUENCE_LENGTH = 80  # from McMahan et al AISTATS 2017


def get_word_dict():
    global word_dict
    if word_dict == None:
        words = [_pad] + CHAR_VOCAB + [_bos] + [_eos]
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
            pad_length = (-len(tokens)) % (max_seq_len + 1)
            tokens += [char_to_id(_pad)] * pad_length
        return (tokens[i:i + max_seq_len + 1]
                for i in range(0, len(tokens), max_seq_len + 1))

    for sen in sentences:
        sequences.extend(to_ids(sen))
    return sequences


class ShakespearNWP(FederatedDataset):
    """Used for next word prediction.
    """

    def __init__(self, root: str, train: bool = True):
        # TODO: 把自动下载数据集的代码添加到这里
        data_file = os.path.join(
            root, DEFAULT_TRAIN_FILE if train else DEFAULT_TEST_FILE)

        data_h5 = h5py.File(data_file, "r")

        client_ids = list(data_h5[_EXAMPLE].keys())

        self.total_parts = len(client_ids)

        self.part_id = 0

        parts_data_list = []
        for client_id in client_ids:
            parts_data_list.append(
                [x.decode('utf-8') for x in np.array(data_h5[_EXAMPLE][client_id][_SNIPPETS][()])])
        self.parts_data_list = parts_data_list

        self.classes = len(get_word_dict()) + 1

    def __len__(self) -> int:
        return len(self.parts_data_list[self.part_id])

    def __getitem__(self, index: int):
        data = preprocess([self.parts_data_list[self.part_id][index]])[0]

        data = torch.tensor(data).reshape(-1)

        x, y = data[:-1], data[-1]
        return x, y

    def total_samples(self):
        return sum([len(x) for x in self.parts_data_list])


class ShakespearNCP(FederatedDataset):
    """Used for next char prediction. FedProx version.
    """

    def __init__(self, root: str, train: bool = True):
        data_file = os.path.join(
            root, DEFAULT_TRAIN_FILE if train else DEFAULT_TEST_FILE)

        with open(data_file, "r") as f:
            data_json = json.load(f)
            # dict
            parts_data_list = data_json[_SNIPPETS]

        self.part_id = 0
        self.total_parts = len(parts_data_list.keys())
        self.parts_name = list(parts_data_list.keys())

        self.parts_data_list = parts_data_list

        self.classes = VOCAB_SIZE

    def __len__(self) -> int:
        part_name = self.parts_name[self.part_id]
        return len(self.parts_data_list[part_name]['x'])

    def __getitem__(self, index: int):
        part_name = self.parts_name[self.part_id]
        data = self.parts_data_list[part_name]
        x, y = word_to_indices(
            data['x'][index]), letter_to_index(data['y'][index])

        return torch.tensor(x), torch.tensor(y)

    def total_samples(self):
        return sum([len(x['x']) for x in self.parts_data_list.values()])
