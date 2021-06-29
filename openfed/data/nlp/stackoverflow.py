import collections
import json
import os

import h5py
import numpy as np
import torch
from openfed.common import logger

from ..dataset import FederatedDataset
from ..utils import *

word_count_file_path = None
word_dict = None
word_list = None
_pad = '<pad>'
_bos = '<bos>'
_eos = '<eos>'
'''
This code follows the steps of preprocessing in tff stackoverflow dataset: 
https://github.com/google-research/federated/blob/master/utils/datasets/stackoverflow_dataset.py
'''
DEFAULT_TRAIN_CLIENTS_NUM = 342477
DEFAULT_TEST_CLIENTS_NUM = 204088
DEFAULT_BATCH_SIZE = 100
DEFAULT_TRAIN_FILE = 'stackoverflow_train.h5'
DEFAULT_TEST_FILE = 'stackoverflow_test.h5'
_EXAMPLE = 'examples'
_TOKENS = 'tokens'
_TITLE = 'title'
_TAGS = 'tags'


DEFAULT_WORD_COUNT_FILE = 'stackoverflow.word_count'
DEFAULT_TAG_COUNT_FILE = 'stackoverflow.tag_count'
word_count_file_path = None
tag_count_file_path = None
word_dict = None
tag_dict = None


def get_word_count_file(data_dir):
    # word_count_file_path
    global word_count_file_path
    if word_count_file_path is None:
        word_count_file_path = os.path.join(data_dir, DEFAULT_WORD_COUNT_FILE)
    return word_count_file_path


def get_tag_count_file(data_dir):
    # tag_count_file_path
    global tag_count_file_path
    if tag_count_file_path is None:
        tag_count_file_path = os.path.join(data_dir, DEFAULT_TAG_COUNT_FILE)
    return tag_count_file_path


def get_most_frequent_words(data_dir=None, vocab_size=10000):
    frequent_words = []
    with open(get_word_count_file(data_dir), 'r') as f:
        frequent_words = [next(f).split()[0] for i in range(vocab_size)]
    return frequent_words


def get_tags(data_dir=None, tag_size=500):
    f = open(get_tag_count_file(data_dir), 'r')
    frequent_tags = json.load(f)
    return list(frequent_tags.keys())[:tag_size]


def get_word_dict(data_dir):
    global word_dict
    if word_dict == None:
        words = get_most_frequent_words(data_dir)
        word_dict = collections.OrderedDict()
        for i, w in enumerate(words):
            word_dict[w] = i
    return word_dict


def get_tag_dict(data_dir):
    global tag_dict
    if tag_dict == None:
        tags = get_tags(data_dir)
        tag_dict = collections.OrderedDict()
        for i, w in enumerate(tags):
            tag_dict[w] = i
    return tag_dict


def preprocess_inputs(sentences, data_dir):

    sentences = [sentence.split(' ') for sentence in sentences]
    vocab_size = len(get_word_dict(data_dir))

    def word_to_id(word):
        word_dict = get_word_dict(data_dir)
        if word in word_dict:
            return word_dict[word]
        else:
            return len(word_dict)

    def to_bag_of_words(sentence):
        tokens = [word_to_id(token) for token in sentence]
        onehot = np.zeros((len(tokens), vocab_size + 1))
        onehot[np.arange(len(tokens)), tokens] = 1
        return np.mean(onehot, axis=0)[:vocab_size]

    return [to_bag_of_words(sentence) for sentence in sentences]


def preprocess_targets(tags, data_dir):

    tags = [tag.split('|') for tag in tags]
    tag_size = len(get_tag_dict(data_dir))

    def tag_to_id(tag):
        tag_dict = get_tag_dict(data_dir)
        if tag in tag_dict:
            return tag_dict[tag]
        else:
            return len(tag_dict)

    def to_bag_of_words(tag):
        tag = [tag_to_id(t) for t in tag]
        onehot = np.zeros((len(tag), tag_size + 1))
        onehot[np.arange(len(tag)), tag] = 1
        return np.sum(onehot, axis=0, dtype=np.float32)  # [:tag_size]

    return [to_bag_of_words(tag) for tag in tags]


def preprocess_input(sentence, data_dir):

    sentence = sentence.split(' ')
    vocab_size = len(get_word_dict(data_dir))

    def word_to_id(word):
        word_dict = get_word_dict(data_dir)
        if word in word_dict:
            return word_dict[word]
        else:
            return len(word_dict)

    def to_bag_of_words(sentence):
        tokens = [word_to_id(token) for token in sentence]
        onehot = np.zeros((len(tokens), vocab_size + 1))
        onehot[np.arange(len(tokens)), tokens] = 1
        return np.mean(onehot, axis=0, dtype=np.float32)[:vocab_size]

    return to_bag_of_words(sentence)


def preprocess_target(tag, data_dir):

    tag = tag.split('|')
    tag_size = len(get_tag_dict(data_dir))

    def tag_to_id(tag):
        tag_dict = get_tag_dict(data_dir)
        if tag in tag_dict:
            return tag_dict[tag]
        else:
            return len(tag_dict)

    def to_bag_of_words(tag):
        tag = [tag_to_id(t) for t in tag]
        onehot = np.zeros((len(tag), tag_size + 1))
        onehot[np.arange(len(tag)), tag] = 1
        return np.sum(onehot, axis=0, dtype=np.float32)[:tag_size]

    return to_bag_of_words(tag)


def get_word_count_file(data_dir):
    # word_count_file_path
    global word_count_file_path
    if word_count_file_path is None:
        word_count_file_path = os.path.join(data_dir, DEFAULT_WORD_COUNT_FILE)
    return word_count_file_path


def get_most_frequent_words(data_dir, vocab_size=10000):
    frequent_words = []
    with open(get_word_count_file(data_dir), 'r') as f:
        frequent_words = [next(f).split()[0] for _ in range(vocab_size)]
    return frequent_words


def get_word_dict(data_dir):
    global word_dict
    if word_dict == None:
        frequent_words = get_most_frequent_words(data_dir)
        words = [_pad] + frequent_words + [_bos] + [_eos]
        word_dict = collections.OrderedDict()
        for i, w in enumerate(words):
            word_dict[w] = i
    return word_dict


def get_word_list(data_dir):
    global word_list
    if word_list == None:
        word_dict = get_word_dict()
        word_list = list(word_dict.keys())
    return word_list


def id_to_word(idx):
    return get_word_list()[idx]


def tokenizer(sentence, data_dir, max_seq_len=20):

    truncated_sentences = sentence.split(' ')[:max_seq_len]

    def word_to_id(word, num_oov_buckets=1):
        word_dict = get_word_dict(data_dir)
        if word in word_dict:
            return word_dict[word]
        else:
            return hash(word) % num_oov_buckets + len(word_dict)

    def to_ids(sentence, num_oov_buckets=1):
        '''
        map list of sentence to list of [idx..] and pad to max_seq_len + 1
        Args:
            num_oov_buckets : The number of out of vocabulary buckets.
            max_seq_len: Integer determining shape of padded batches.
        '''
        tokens = [word_to_id(token) for token in sentence]
        if len(tokens) < max_seq_len:
            tokens = tokens + [word_to_id(_eos)]
        tokens = [word_to_id(_bos)] + tokens
        if len(tokens) < max_seq_len + 1:
            tokens += [word_to_id(_pad)] * (max_seq_len + 1 - len(tokens))
        return tokens

    return to_ids(truncated_sentences)


def split(dataset):
    ds = np.array(dataset)
    x = ds[:, :-1]
    y = ds[:, -1]
    return x, y


class StackOverFlow(FederatedDataset):
    def __init__(self, root: str, train: bool = True, download: bool = True, transform=None):

        data_file = os.path.join(
            root, DEFAULT_TRAIN_FILE if train else DEFAULT_TEST_FILE)
        if not os.path.isfile(data_file):
            if download:
                urls = [
                    'https://fedml.s3-us-west-1.amazonaws.com/stackoverflow.tag_count.tar.bz2',
                    'https://fedml.s3-us-west-1.amazonaws.com/stackoverflow.word_count.tar.bz2',
                    'https://fedml.s3-us-west-1.amazonaws.com/stackoverflow.tar.bz2',
                    'https://fedml.s3-us-west-1.amazonaws.com/stackoverflow_nwp.pkl', ]
                for url in urls:
                    logger.debug(f"Download dataset from {url} to {root}")
                    if wget_https(url, root):
                        if url.endswith(".bz2"):
                            if tar_xvf(os.path.join(root, url.split("/")[-1]), output_dir=root):
                                logger.debug("Downloaded.")
                    else:
                        raise RuntimeError("Download dataset failed.")
            else:
                raise FileNotFoundError(f"{data_file} not exists.")

        self.data_file = data_file
        self.root = root
        self.transform = transform


class StackOverFlowTP(StackOverFlow):
    """tag prediction.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        with h5py.File(self.data_file, "r") as data_h5:
            client_ids = list(data_h5[_EXAMPLE].keys())

        self.total_parts = len(client_ids)
        self.parts_name = client_ids

        self.classes = len(get_tag_dict(self.root))

    def __len__(self) -> int:
        with h5py.File(self.data_file, 'r') as data_h5:
            part_name = self.parts_name[self.part_id]
            return len(data_h5[_EXAMPLE][part_name][_TAGS][()])

    def __getitem__(self, index: int):
        with h5py.File(self.data_file, "r") as data_h5:
            part_name = self.parts_name[self.part_id]
            raw_token = data_h5[_EXAMPLE][part_name][_TOKENS][()][index].decode(
                'utf-8')
            raw_title = data_h5[_EXAMPLE][part_name][_TITLE][()][index].decode(
                'utf-8')
            sample = ' '.join([raw_token, raw_title])
            tag = data_h5[_EXAMPLE][part_name][_TAGS][()
                                                      ][index].decode('utf-8')

            sample = preprocess_input(sample, self.root)
            tag = preprocess_target(tag, self.root)

        if self.transform:
            sample, tag = self.transform(sample), self.transform(tag)

        return sample, tag

    def total_samples(self):
        samples = []
        for part_name in self.parts_name:
            with h5py.File(self.data_file, 'r') as data_h5:
                samples.append(len(data_h5[_EXAMPLE][part_name][_TAGS][()]))

        return sum(samples)


class StackOverFlowNWP(StackOverFlow):
    """next work prediction.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        with h5py.File(self.data_file, "r") as data_h5:
            client_ids = list(data_h5[_EXAMPLE].keys())

        with h5py.File(self.data_file, "r") as data_h5:
            client_ids = list(data_h5[_EXAMPLE].keys())

        self.total_parts = len(client_ids)
        self.parts_name = client_ids

        self.classes = len(get_word_dict(self.root)) + 1

    def __len__(self) -> int:
        with h5py.File(self.data_file, 'r') as data_h5:
            part_name = self.parts_name[self.part_id]
            return len(data_h5[_EXAMPLE][part_name][_TOKENS][()])

    def __getitem__(self, index: int):
        with h5py.File(self.data_file, "r") as data_h5:
            part_name = self.parts_name[self.part_id]
            raw_token = data_h5[_EXAMPLE][part_name][_TOKENS][()][index].decode(
                'utf-8')
            sample = tokenizer(raw_token, self.root)
        x, y = sample[:-1], torch.tensor(sample[1:])
        if self.transform:
            x, y = self.transform(x), self.transform(y)
        return x, y

    def total_samples(self):
        samples = []
        for part_name in self.parts_name:
            with h5py.File(self.data_file, 'r') as data_h5:
                samples.append(len(data_h5[_EXAMPLE][part_name][_TOKENS][()]))

        return sum(samples)
