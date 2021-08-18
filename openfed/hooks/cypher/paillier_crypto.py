import time
from typing import Dict, Union

import torch
from torch import Tensor

from .cypher import Cypher
from openfed.utils import openfed_class_fmt


class PublicKey(object):
    def __init__(self, A, P, n_lwe, bits, l, bound):
        self.A = A
        self.P = P
        self.n_lwe = n_lwe
        self.bits = bits
        self.bits_safe = bits - 8
        self.l = l
        self.bound = bound

        self.p = 2**self.bits + 1
        self.q = 2**self.bits

    def __repr__(self):
        return openfed_class_fmt.format(
            class_name="PublicKey",
            description=
            f'n_lwe: {self.n_lwe}, bits: {self.bits}, bits_safe: {self.bits_safe}, '
            f'l: {self.l}, bound: {self.bound}, p: {self.p}, q: {self.q}')


class PrivateKey(object):
    def __init__(self, S, n_lwe, bits, l, bound):
        self.S = S
        self.n_lwe = n_lwe
        self.bits = bits
        self.bits_safe = bits - 8
        self.l = l
        self.bound = bound

        self.p = 2**self.bits + 1
        self.q = 2**self.bits

    def __repr__(self):
        return openfed_class_fmt.format(
            class_name="PrivateKey",
            description=
            f'n_lwe: {self.n_lwe}, bits: {self.bits}, bits_safe: {self.bits_safe}, '
            f'l: {self.l}, bound: {self.bound}, p: {self.p}, q: {self.q}')


class Ciphertext(object):
    def __init__(self, c1, c2):
        self.c1 = c1
        self.c2 = c2

    def __repr__(self):
        return 'Ciphertext({}, {})'.format(self.c1, self.c2)

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
            l: int = 2**6,
            bound: int = 2**3):
    p = 2**bits + 1
    q = 2**bits
    R = get_discrete_gaussian_random_matrix(n_lwe, l, bits)
    S = get_discrete_gaussian_random_matrix(n_lwe, l, bits)
    A = get_uniform_random_matrix(n_lwe, n_lwe, q)

    P = p * R - torch.matmul(A, S)
    return PublicKey(A, P, n_lwe, bits, l,
                     bound), PrivateKey(S, n_lwe, bits, l, bound)


def enc(public_key: PublicKey, m) -> Ciphertext:
    e1 = get_discrete_gaussian_random_vector(public_key.n_lwe,
                                             public_key.bits).to(m)
    e2 = get_discrete_gaussian_random_vector(public_key.n_lwe,
                                             public_key.bits).to(m)
    e3 = get_discrete_gaussian_random_vector(public_key.l,
                                             public_key.bits).to(m)

    A = public_key.A.to(m)
    P = public_key.P.to(m)

    if len(m) % public_key.l != 0:
        m = torch.cat(
            (m, torch.zeros(public_key.l - len(m) % public_key.l).type_as(m)),
            0)

    m = m.reshape(-1, public_key.l)

    c1 = torch.matmul(e1, A) + public_key.p * e2
    c2 = torch.matmul(e1, P) + public_key.p * e3
    c2 = c2.unsqueeze(dim=0) + m

    return Ciphertext(c1, c2)


def dec(private_key, c) -> Tensor:
    S = private_key.S.to(c.c1)
    return (torch.matmul(c.c1, S).unsqueeze(0) +
            c.c2).reshape(-1) % private_key.p


def float_to_long(public_key: PublicKey, tensor):
    return ((tensor + public_key.bound) * 2**(public_key.bits_safe)).long()


def long_to_float(private_key: PrivateKey, tensor, denominator):
    return tensor.float() / (
        2**private_key.bits_safe) / denominator - private_key.bound


class PaillierCrypto(Cypher):
    """
        NOTE: Paillier is only be used in follower. It must pair with PaillierAggregator.
    """
    def __init__(self, public_key: Union[str, PublicKey]):
        """
        Args: 
            public_key: PublicKey or the path to load it.
        """
        super().__init__()
        if isinstance(public_key, str):
            public_key = torch.load(public_key)
        self.public_key: PublicKey = public_key  # type: ignore

    def encrypt(self, key: Union[str, Tensor],
                value: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """<key, value> pair in the package before transfer to the other end.
        """
        encrypt_value = dict()
        for k, v in value.items():
            tmp = float_to_long(self.public_key, v).view(-1)
            enc_v = enc(self.public_key, tmp)
            encrypt_value[f"{k}_c1"] = enc_v.c1
            encrypt_value[f"{k}_c2"] = enc_v.c2
        return encrypt_value

    def decrypt(self, key: Union[str, Tensor],
                value: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """<key, value> pair in the package received from the other end.
        """
        return value


if __name__ == "__main__":
    st = time.time()
    l = 2**3
    public_key, private_key = key_gen(l=l)
    print("KeyGen Time: %.6f bits" % (time.time() - st))

    _m1 = torch.randn(l)
    _m2 = torch.randn(l)

    m1 = float_to_long(public_key, _m1)
    m2 = float_to_long(public_key, _m2)

    st = time.time()
    c1 = enc(public_key, m1)
    c2 = enc(public_key, m2)
    print("Encrypt Time: %.6f ms/op" % ((time.time() - st) * 1000 / (2 * l)))

    st = time.time()
    c = c1 + c2
    print("Add Time: %.6f ms/op" % ((time.time() - st) * 1000 / l))

    st = time.time()
    m = dec(private_key, c)
    print("Decrypt Time: %.6f ms/op" % ((time.time() - st) * 1000 / l))

    acc = torch.mean(torch.eq(m, m1 + m2).float())
    print('Accuracy: %.2f%%' % (acc * 100))

    m = long_to_float(private_key, m, 2)

    acc = torch.mean((torch.abs(m - (_m1 + _m2) / 2) < 1e-5).float())
    print('Accuracy: %.2f%%' % (acc * 100))
