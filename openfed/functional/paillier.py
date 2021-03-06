# @Author            : FederalLab
# @Date              : 2021-09-25 16:53:10
# @Last Modified by  : Chen Dengsheng
# @Last Modified time: 2021-09-25 16:53:10
# Copyright (c) FederalLab. All rights reserved.
from abc import abstractmethod
from typing import Union

import torch
from torch import Tensor

from openfed.core.const import DefaultMaintainer
from openfed.utils import FMT, tablist


class Key(object):

    def __init__(self, n_lwe, bits, lines, bound):
        self.n_lwe = n_lwe
        self.bits = bits
        self.bits_safe = bits - 8
        self.lines = lines
        self.bound = bound

        self.p = 2**self.bits + 1
        self.q = 2**self.bits

    def save(self, key_file):
        torch.save(self.state_dict(), key_file)

    def __repr__(self):
        head = ['n_lwe', 'bits', 'bits_safe', 'lines', 'bound', 'p', 'q']
        data = [
            self.n_lwe, self.bits, self.bits_safe, self.lines, self.bound,
            self.p, self.q
        ]
        description = tablist(head, data, force_in_one_row=True)

        return FMT.openfed_class_fmt.format(
            class_name=self.__class__.__name__, description=description)

    @abstractmethod
    def state_dict(self):
        ...


class PublicKey(Key):

    def __init__(self, A, P, n_lwe, bits, lines, bound, **kwargs):
        super().__init__(n_lwe, bits, lines, bound)
        self.A = A
        self.P = P

    def state_dict(self):
        return dict(
            A=self.A,
            P=self.P,
            n_lwe=self.n_lwe,
            bits=self.bits,
            bits_safe=self.bits_safe,
            lines=self.lines,
            bound=self.bound,
            p=self.p,
            q=self.q,
        )

    @classmethod
    def load(cls, key_file):
        return PublicKey(**torch.load(key_file))


class PrivateKey(Key):

    def __init__(self, S, n_lwe, bits, lines, bound, **kwargs):
        super().__init__(n_lwe, bits, lines, bound)
        self.S = S

    def state_dict(self):
        return dict(
            S=self.S,
            n_lwe=self.n_lwe,
            bits=self.bits,
            bits_safe=self.bits_safe,
            lines=self.lines,
            bound=self.bound,
            p=self.p,
            q=self.q,
        )

    @classmethod
    def load(cls, key_file):
        return PrivateKey(**torch.load(key_file))


class Ciphertext(object):

    def __init__(self, c1, c2):
        self.c1 = c1
        self.c2 = c2

    def __repr__(self):
        description = f'c1: {self.c1}, c2: {self.c2}'
        return FMT.openfed_class_fmt.format(
            class_name=self.__class__.__name__, description=description)

    def __add__(self, other):
        return Ciphertext(self.c1 + other.c1, self.c2 + other.c2)


def get_discrete_gaussian_random_matrix(m, n, bits):
    return torch.normal(0, bits, (m, n)).long()


def get_discrete_gaussian_random_vector(n, bits):
    return torch.normal(0, bits, (n, )).long()


def get_uniform_random_matrix(m, n, q):
    return torch.randint(-q // 2 + 1, q // 2, (m, n)).long()


def key_gen(n_lwe: int = 3000,
            bits: int = 32,
            lines: int = 2**6,
            bound: int = 2**3):
    p = 2**bits + 1
    q = 2**bits
    R = get_discrete_gaussian_random_matrix(n_lwe, lines, bits)
    S = get_discrete_gaussian_random_matrix(n_lwe, lines, bits)
    A = get_uniform_random_matrix(n_lwe, n_lwe, q)

    P = p * R - torch.matmul(A, S)
    return PublicKey(A, P, n_lwe, bits, lines,
                     bound), PrivateKey(S, n_lwe, bits, lines, bound)


def paillier_enc(public_key: PublicKey, m: Tensor) -> Ciphertext:
    e1 = get_discrete_gaussian_random_vector(public_key.n_lwe,
                                             public_key.bits).to(m)
    e2 = get_discrete_gaussian_random_vector(public_key.n_lwe,
                                             public_key.bits).to(m)
    e3 = get_discrete_gaussian_random_vector(public_key.lines,
                                             public_key.bits).to(m)

    A = public_key.A.to(m)
    P = public_key.P.to(m)

    if len(m) % public_key.lines != 0:
        m = torch.cat(
            (m, torch.zeros(public_key.lines -
                            len(m) % public_key.lines).type_as(m)), 0)

    m = m.reshape(-1, public_key.lines)

    c1 = torch.matmul(e1, A) + public_key.p * e2
    c2 = torch.matmul(e1, P) + public_key.p * e3
    c2 = c2.unsqueeze(dim=0) + m

    return Ciphertext(c1, c2)


def paillier_dec(private_key: PrivateKey, c: Ciphertext) -> Tensor:
    S = private_key.S.to(c.c1)
    return (torch.matmul(c.c1, S).unsqueeze(0) +
            c.c2).reshape(-1) % private_key.p


def float_to_long(public_key: PublicKey, tensor: Tensor):
    return ((tensor + public_key.bound) * 2**(public_key.bits_safe)).long()


def long_to_float(private_key: PrivateKey, tensor: Tensor, denominator: float):
    return tensor.float() / (
        2**private_key.bits_safe) / denominator - private_key.bound


def paillier_package(public_key: Union[str, PublicKey]):
    _default_maintainer = DefaultMaintainer._default_maintainer

    assert _default_maintainer, \
        'Define a maintainer and use `with maintainer` context.'

    if isinstance(public_key, str):
        public_key = PublicKey.load(public_key)

    if _default_maintainer.collaborator:

        def package(state, p):
            paillier_state = dict()
            for k, v in state.items():
                if v is not None:
                    flat_v = float_to_long(
                        public_key,  # type: ignore
                        v).view(-1)
                    ciphertext = paillier_enc(
                        public_key,  # type: ignore
                        flat_v)
                    paillier_state[f'{k}_c1'] = ciphertext.c1
                    paillier_state[f'{k}_c2'] = ciphertext.c2
            return paillier_state

        _default_maintainer.register_package_hook(
            nice=90, package_hook=package)
