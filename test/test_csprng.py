# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import torch
from scipy import stats
import numpy as np
import math
import random
import time
import os
from Crypto.Cipher import AES
from Crypto.Util import Counter

try:
    import torchcsprng as csprng
except ImportError:
    raise RuntimeError("CSPRNG not available")

IS_SANDCASTLE = os.getenv('SANDCASTLE') == '1' or os.getenv('TW_JOB_USER') == 'sandcastle'
IS_FBCODE = os.getenv('PYTORCH_TEST_FBCODE') == '1'


def to_numpy(t, dtype=torch.float):
    if t.dtype == torch.bfloat16:
        t = t.to(dtype)
    return t.numpy()


def to_bytes(t):
    if t.dtype == torch.bfloat16:
        t = t.view(torch.int16)
    return t.cpu().numpy().view(np.int8)


class TestCSPRNG(unittest.TestCase):

    all_generators = [
        csprng.create_random_device_generator(),
        csprng.create_random_device_generator('/dev/urandom'),
        csprng.create_mt19937_generator(),
        csprng.create_mt19937_generator(42)
    ]

    int_dtypes = [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]

    standard_fp_dtypes = [torch.float, torch.double]

    non_standard_fp_dtypes = [torch.half, torch.bfloat16]

    fp_dtypes = standard_fp_dtypes + non_standard_fp_dtypes

    num_dtypes = int_dtypes + fp_dtypes

    all_dtypes = num_dtypes + [torch.bool]

    size = 1000

    all_devices = ['cpu', 'cuda'] if (torch.cuda.is_available() and csprng.supports_cuda()) else ['cpu']

    def test_random_kstest(self):
        for device in self.all_devices:
            for gen in self.all_generators:
                for dtype in self.num_dtypes:
                    if dtype == torch.float:
                        to_inc = 2**24
                    elif dtype == torch.double:
                        to_inc = 2**53
                    elif dtype == torch.half:
                        to_inc = 2**11
                    elif dtype == torch.bfloat16:
                        to_inc = 2**8
                    else:
                        to_inc = torch.iinfo(dtype).max

                    t = torch.empty(self.size, dtype=dtype, device=device).random_(generator=gen)
                    res = stats.kstest(to_numpy(t.cpu()), stats.randint.cdf, args=(0, to_inc))
                    self.assertTrue(res.statistic < 0.1)

    no_cuda = not torch.cuda.is_available() or not csprng.supports_cuda()

    no_cuda_message = "CUDA is not available or csprng was not compiled with CUDA support"

    @unittest.skipIf(no_cuda, no_cuda_message)
    def test_random_cpu_vs_cuda(self):
        for dtype in self.num_dtypes:
            gen = csprng.create_mt19937_generator(42)
            cpu_t = torch.empty(self.size, dtype=dtype, device='cpu').random_(generator=gen)
            gen = csprng.create_mt19937_generator(42)
            cuda_t = torch.empty(self.size, dtype=dtype, device='cuda').random_(generator=gen)
            self.assertTrue((cpu_t == cuda_t.cpu()).all())

    def test_random_to_kstest(self):
        to_ = 42
        for device in self.all_devices:
            for gen in self.all_generators:
                for dtype in self.num_dtypes:
                    t = torch.zeros(self.size, dtype=dtype, device=device).random_(to_, generator=gen)
                    res = stats.kstest(to_numpy(t.cpu()), stats.randint.cdf, args=(0, to_))
                    self.assertTrue(res.statistic < 0.1)

    @unittest.skipIf(no_cuda, no_cuda_message)
    def test_random_to_cpu_vs_cuda(self):
        to_ = 42
        for dtype in self.num_dtypes:
            gen = csprng.create_mt19937_generator(42)
            cpu_t = torch.zeros(self.size, dtype=dtype, device='cpu').random_(to_, generator=gen)
            gen = csprng.create_mt19937_generator(42)
            cuda_t = torch.zeros(self.size, dtype=dtype, device='cuda').random_(to_, generator=gen)
            self.assertTrue((cpu_t == cuda_t.cpu()).all())

    def test_random_from_to_kstest(self):
        for device in self.all_devices:
            for gen in self.all_generators:
                for dtype in self.num_dtypes:
                    for from_ in [0, 24, 42]:
                        for to_ in [42, 99, 123]:
                            if from_ < to_:
                                t = torch.zeros(self.size, dtype=dtype, device=device).random_(from_, to_, generator=gen)
                                res = stats.kstest(to_numpy(t.cpu()), stats.randint.cdf, args=(from_, to_))
                                self.assertTrue(res.statistic < 0.2)

    @unittest.skipIf(no_cuda, no_cuda_message)
    def test_random_from_to_cpu_vs_cuda(self):
        for dtype in self.num_dtypes:
            for from_ in [0, 24, 42]:
                for to_ in [42, 99, 123]:
                    if from_ < to_:
                        gen = csprng.create_mt19937_generator(42)
                        cpu_t = torch.zeros(self.size, dtype=dtype, device='cpu').random_(from_, to_, generator=gen)
                        gen = csprng.create_mt19937_generator(42)
                        cuda_t = torch.zeros(self.size, dtype=dtype, device='cuda').random_(from_, to_, generator=gen)
                        self.assertTrue((cpu_t == cuda_t.cpu()).all())

    def test_random_bool(self):
        for device in self.all_devices:
            for gen in self.all_generators:
                t = torch.empty(self.size, dtype=torch.bool, device=device)

                t.fill_(False)
                t.random_(generator=gen)
                self.assertEqual(t.min(), False)
                self.assertEqual(t.max(), True)
                self.assertTrue(0.4 < (t.eq(True)).to(torch.int).sum().item() / self.size < 0.6)

                t.fill_(True)
                t.random_(generator=gen)
                self.assertEqual(t.min(), False)
                self.assertEqual(t.max(), True)
                self.assertTrue(0.4 < (t.eq(True)).to(torch.int).sum().item() / self.size < 0.6)

    @unittest.skipIf(no_cuda, no_cuda_message)
    def test_random_bool_cpu_vs_cuda(self):
        gen = csprng.create_mt19937_generator(42)
        cpu_t = torch.empty(self.size, dtype=torch.bool, device='cpu').random_(generator=gen)
        gen = csprng.create_mt19937_generator(42)
        cuda_t = torch.empty(self.size, dtype=torch.bool, device='cuda').random_(generator=gen)
        self.assertTrue((cpu_t == cuda_t.cpu()).all())

    def test_uniform_kstest(self):
        for device in self.all_devices:
            for gen in self.all_generators:
                for dtype in self.fp_dtypes:
                    for from_ in [-42, 0, 4.2]:
                        for to_ in [-4.2, 0, 42]:
                            if to_ > from_:
                                t = torch.empty(self.size, dtype=dtype, device=device).uniform_(from_, to_, generator=gen)
                                res = stats.kstest(to_numpy(t.cpu(), torch.double), 'uniform', args=(from_, (to_ - from_)))
                                self.assertTrue(res.statistic < 0.1)

    @unittest.skipIf(no_cuda, no_cuda_message)
    def test_uniform_cpu_vs_cuda(self):
        for dtype in self.fp_dtypes:
            for from_ in [-42, 0, 4.2]:
                for to_ in [-4.2, 0, 42]:
                    if to_ > from_:
                        gen = csprng.create_mt19937_generator(42)
                        cpu_t = torch.empty(self.size, dtype=dtype, device='cpu').uniform_(from_, to_, generator=gen)
                        gen = csprng.create_mt19937_generator(42)
                        cuda_t = torch.empty(self.size, dtype=dtype, device='cuda').uniform_(from_, to_, generator=gen)
                        self.assertTrue(torch.allclose(cpu_t, cuda_t.cpu(), 1e-9))

    def test_normal_kstest(self):
        for device in self.all_devices:
            for gen in self.all_generators:
                for dtype in self.fp_dtypes:
                    for mean in [-3, 0, 7]:
                        for std in [1, 5, 7]:
                            t = torch.empty(self.size, dtype=dtype, device=device).normal_(mean=mean, std=std, generator=gen)
                            res = stats.kstest(to_numpy(t.cpu(), torch.double), 'norm', args=(mean, std))
                            self.assertTrue(res.statistic < 0.1)

    @unittest.skipIf(no_cuda, no_cuda_message)
    def test_normal_cpu_vs_cuda(self):
        for dtype in self.fp_dtypes:
            for mean in [-3, 0, 7]:
                for std in [1, 5, 7]:
                    gen = csprng.create_mt19937_generator(42)
                    cpu_t = torch.empty(self.size, dtype=dtype, device='cpu').normal_(mean=mean, std=std, generator=gen)
                    gen = csprng.create_mt19937_generator(42)
                    cuda_t = torch.empty(self.size, dtype=dtype, device='cuda').normal_(mean=mean, std=std, generator=gen)
                    self.assertTrue(torch.allclose(cpu_t, cuda_t.cpu(), 1e-9))

    def test_log_normal_kstest(self):
        for device in self.all_devices:
            for gen in self.all_generators:
                for dtype in self.fp_dtypes:
                    for mean in [-3, 0, 7]:
                        for std in [1, 5, 7]:
                            t = torch.empty(self.size, dtype=dtype, device=device).log_normal_(mean=mean, std=std, generator=gen)
                            res = stats.kstest(to_numpy(t.cpu(), torch.double), 'lognorm', args=(std, 0, math.exp(mean)))
                            if dtype in [torch.half, torch.bfloat16]:
                                self.assertTrue(res.statistic < 0.4)
                            else:
                                self.assertTrue(res.statistic < 0.1)

    @unittest.skipIf(no_cuda, no_cuda_message)
    def test_log_normal_cpu_vs_cuda(self):
        for dtype in self.fp_dtypes:
            for mean in [-3, 0, 7]:
                for std in [1, 5, 7]:
                    gen = csprng.create_mt19937_generator(42)
                    cpu_t = torch.empty(self.size, dtype=dtype, device='cpu').log_normal_(mean=mean, std=std, generator=gen)
                    gen = csprng.create_mt19937_generator(42)
                    cuda_t = torch.empty(self.size, dtype=dtype, device='cuda').log_normal_(mean=mean, std=std, generator=gen)
                    self.assertTrue(torch.allclose(cpu_t, cuda_t.cpu(), 1e-4, equal_nan=True))

    def test_exponential_kstest(self):
        for device in self.all_devices:
            for gen in self.all_generators:
                for dtype in self.fp_dtypes:
                    for lambd in [0.5, 1.0, 5.0]:
                        t = torch.empty(self.size, dtype=dtype, device=device).exponential_(lambd=lambd, generator=gen)
                        res = stats.kstest(to_numpy(t.cpu(), torch.double), 'expon', args=(0, 1 / lambd,))
                        self.assertTrue(res.statistic < 0.1)

    @unittest.skipIf(no_cuda, no_cuda_message)
    @unittest.skip("https://github.com/pytorch/pytorch/issues/38662")
    def test_exponential_cpu_vs_cuda(self):
        for dtype in self.fp_dtypes:
            for lambd in [0.5, 1.0, 5.0]:
                gen = csprng.create_mt19937_generator(42)
                cpu_t = torch.empty(self.size, dtype=dtype, device='cpu').exponential_(lambd=lambd, generator=gen)
                gen = csprng.create_mt19937_generator(42)
                cuda_t = torch.empty(self.size, dtype=dtype, device='cuda').exponential_(lambd=lambd, generator=gen)
                self.assertTrue(torch.allclose(cpu_t, cuda_t.cpu(), 1e-9))

    def test_cauchy_kstest(self):
        for device in self.all_devices:
            for gen in self.all_generators:
                for dtype in self.fp_dtypes:
                    for median in [-10, 0, 50]:
                        for sigma in [0.5, 1.0, 10.0]:
                            t = torch.empty(self.size, dtype=dtype, device=device).cauchy_(median=median, sigma=sigma, generator=gen)
                            res = stats.kstest(to_numpy(t.cpu(), torch.double), 'cauchy', args=(median, sigma))
                            if dtype in [torch.half, torch.bfloat16]:
                                self.assertTrue(res.statistic < 0.4)
                            else:
                                self.assertTrue(res.statistic < 0.1)

    @unittest.skipIf(no_cuda, no_cuda_message)
    def test_cauchy_cpu_vs_cuda(self):
        for dtype in self.fp_dtypes:
            for median in [-10, 0, 50]:
                for sigma in [0.5, 1.0, 10.0]:
                    gen = csprng.create_mt19937_generator(42)
                    cpu_t = torch.empty(self.size, dtype=dtype, device='cpu').cauchy_(median=median, sigma=sigma, generator=gen)
                    gen = csprng.create_mt19937_generator(42)
                    cuda_t = torch.empty(self.size, dtype=dtype, device='cuda').cauchy_(median=median, sigma=sigma, generator=gen)
                    self.assertTrue(torch.allclose(cpu_t, cuda_t.cpu(), 1e-9))

    def test_geometric(self):
        for device in self.all_devices:
            for gen in self.all_generators:
                for dtype in self.fp_dtypes:
                    for p in [0.2, 0.5, 0.8]:
                        t = torch.empty(self.size, dtype=dtype, device=device).geometric_(p=p, generator=gen)
                        # actual = np.histogram(t.cpu().to(torch.double), np.arange(1, 100))[0]
                        # expected = stats.geom(p).pmf(np.arange(1, 99)) * self.size
                        # res = stats.chisquare(actual, expected)
                        # self.assertAlmostEqual(res.pvalue, 1.0, delta=0.5) TODO https://github.com/pytorch/csprng/issues/7

    @unittest.skipIf(no_cuda, no_cuda_message)
    def test_geometric_cpu_vs_cuda(self):
        for dtype in self.fp_dtypes:
            for p in [0.2, 0.5, 0.8]:
                gen = csprng.create_mt19937_generator(42)
                cpu_t = torch.empty(self.size, dtype=dtype, device='cpu').geometric_(p=p, generator=gen)
                gen = csprng.create_mt19937_generator(42)
                cuda_t = torch.empty(self.size, dtype=dtype, device='cuda').geometric_(p=p, generator=gen)
                self.assertTrue(torch.allclose(cpu_t, cuda_t.cpu(), 1e-9, equal_nan=True))

    def test_non_contiguous_vs_contiguous(self):
        size = 10
        for device in self.all_devices:
            for dtype in self.all_dtypes:
                for i in range(10):
                    t = torch.zeros([size, size, size], dtype=dtype, device=device)
                    x1 = random.randrange(0, size)
                    y1 = random.randrange(0, size)
                    z1 = random.randrange(0, size)
                    x2 = random.randrange(x1 + 1, max(x1 + 2, size))
                    y2 = random.randrange(y1 + 1, max(y1 + 2, size))
                    z2 = random.randrange(z1 + 1, max(z1 + 2, size))
                    maybe_non_contiguous = t[x1:x2, y1:y2, z1:z2]
                    assert(maybe_non_contiguous.numel() > 0)

                    if not maybe_non_contiguous.is_contiguous():
                        seed = random.randrange(1000)

                        non_contiguous = maybe_non_contiguous
                        gen = csprng.create_mt19937_generator(seed)
                        non_contiguous.random_(generator=gen)

                        contiguous = torch.zeros_like(non_contiguous)
                        gen = csprng.create_mt19937_generator(seed)
                        contiguous.random_(generator=gen)

                        assert(contiguous.is_contiguous())
                        self.assertTrue((non_contiguous == contiguous).all())

                        for x in range(0, size):
                            for y in range(0, size):
                                for z in range(0, size):
                                    if not x1 <= x < x2 and not y1 <= y < y2 and not z1 <= z < z2:
                                        self.assertTrue(t[x, y, z] == 0)

    @unittest.skipIf(IS_SANDCASTLE or IS_FBCODE, "Does not work on Sandcastle")
    @unittest.skipIf(torch.get_num_threads() < 2, "requires multithreading CPU")
    def test_cpu_parallel(self):
        urandom_gen = csprng.create_random_device_generator('/dev/urandom')

        def measure(size):
            t = torch.empty(size, dtype=torch.float32, device='cpu')
            start = time.time()
            for i in range(20):
                t.normal_(generator=urandom_gen)
            finish = time.time()
            return finish - start

        time_for_1K = measure(1000)
        time_for_1M = measure(1000000)
        # Pessimistic check that parallel execution gives >= 1.5 performance boost
        self.assertTrue(time_for_1M/time_for_1K < 1000 / 1.5)

    @unittest.skipIf(IS_SANDCASTLE or IS_FBCODE, "Does not work on Sandcastle")
    def test_version(self):
        self.assertTrue(csprng.__version__)
        self.assertTrue(csprng.git_version)

    def test_randperm(self):
        for device in self.all_devices:
            for gen in self.all_generators:
                for dtype in self.int_dtypes:
                    for size in range(0, 20):
                        expected = torch.arange(size, dtype=dtype, device=device)

                        actual = torch.randperm(size, dtype=dtype, device=device, generator=gen)

                        actual_out = torch.empty(1, dtype=dtype, device=device)
                        torch.randperm(size, out=actual_out, generator=gen)

                        if size >= 10:
                            self.assertTrue(not torch.allclose(expected, actual))
                            self.assertTrue(not torch.allclose(expected, actual_out))

                        actual = actual.sort()[0]
                        actual_out = actual.sort()[0]

                        self.assertTrue(torch.allclose(expected, actual))
                        self.assertTrue(torch.allclose(expected, actual_out))

    def test_encrypt_decrypt(self):
        key_size_bytes = 16
        block_size_bytes = 16

        def sizeof(dtype):
            if dtype == torch.bool:
                return 1
            elif dtype.is_floating_point:
                return torch.finfo(dtype).bits // 8
            else:
                return torch.iinfo(dtype).bits // 8

        def pad(data, pad_size):
            if len(data) % pad_size == 0:
                return data
            length = pad_size - (len(data) % pad_size)
            return data + bytes([0])*length

        def create_aes(m, k):
            if m == "ecb":
                return AES.new(k.tobytes(), AES.MODE_ECB)
            elif m == "ctr":
                ctr = Counter.new(AES.block_size * 8, initial_value=0, little_endian=True)
                return AES.new(k.tobytes(), AES.MODE_CTR, counter=ctr)
            else:
                return None

        for key_dtype in self.all_dtypes:
            key_size = key_size_bytes // sizeof(key_dtype)
            key = torch.empty(key_size, dtype=key_dtype).random_()
            key_np = to_bytes(key)
            for initial_dtype in self.all_dtypes:
                for initial_size in [0, 4, 8, 15, 16, 23, 42]:
                    initial = torch.empty(initial_size, dtype=initial_dtype).random_()
                    initial_np = to_bytes(initial)
                    initial_size_bytes = initial_size * sizeof(initial_dtype)
                    for encrypted_dtype in self.all_dtypes:
                        encrypted_size = (initial_size_bytes + block_size_bytes - 1) // block_size_bytes * block_size_bytes // sizeof(encrypted_dtype)
                        encrypted = torch.zeros(encrypted_size, dtype=encrypted_dtype)
                        for decrypted_dtype in self.all_dtypes:
                            decrypted_size = (initial_size_bytes + sizeof(decrypted_dtype) - 1) // sizeof(decrypted_dtype)
                            decrypted = torch.zeros(decrypted_size, dtype=decrypted_dtype)
                            for mode in ["ecb", "ctr"]:
                                for device in self.all_devices:
                                    key = key.to(device)
                                    initial = initial.to(device)
                                    encrypted = encrypted.to(device)
                                    decrypted = decrypted.to(device)

                                    csprng.encrypt(initial, encrypted, key, "aes128", mode)
                                    encrypted_np = to_bytes(encrypted)

                                    aes = create_aes(mode, key_np)

                                    encrypted_expected = np.frombuffer(aes.encrypt(pad(initial_np.tobytes(), block_size_bytes)), dtype=np.int8)
                                    self.assertTrue(np.array_equal(encrypted_np, encrypted_expected))

                                    csprng.decrypt(encrypted, decrypted, key, "aes128", mode)
                                    decrypted_np = to_bytes(decrypted)[:initial_size_bytes]

                                    aes = create_aes(mode, key_np)

                                    decrypted_expected = np.frombuffer(aes.decrypt(pad(encrypted_np.tobytes(), block_size_bytes)), dtype=np.int8)[:initial_size_bytes]
                                    self.assertTrue(np.array_equal(decrypted_np, decrypted_expected))

                                    self.assertTrue(np.array_equal(initial_np, decrypted_np))

    def test_encrypt_decrypt_inplace(self):
        key_size_bytes = 16

        def sizeof(dtype):
            if dtype == torch.bool:
                return 1
            elif dtype.is_floating_point:
                return torch.finfo(dtype).bits // 8
            else:
                return torch.iinfo(dtype).bits // 8

        def create_aes(m, k):
            if m == "ecb":
                return AES.new(k.tobytes(), AES.MODE_ECB)
            elif m == "ctr":
                ctr = Counter.new(AES.block_size * 8, initial_value=0, little_endian=True)
                return AES.new(k.tobytes(), AES.MODE_CTR, counter=ctr)
            else:
                return None

        for key_dtype in self.all_dtypes:
            key_size = key_size_bytes // sizeof(key_dtype)
            key = torch.empty(key_size, dtype=key_dtype).random_()
            key_np = to_bytes(key)
            for initial_dtype in self.all_dtypes:
                for initial_size_bytes in [0, 16, 256]:
                    initial_size = initial_size_bytes // sizeof(initial_dtype)
                    initial = torch.empty(initial_size, dtype=initial_dtype).random_()
                    initial_np = to_bytes(initial)
                    initial_np_copy = np.copy(initial_np)
                    for mode in ["ecb", "ctr"]:
                        for device in self.all_devices:
                            key = key.to(device)
                            initial = initial.to(device)

                            csprng.encrypt(initial, initial, key, "aes128", mode)
                            encrypted_np = to_bytes(initial)
                            aes = create_aes(mode, key_np)
                            encrypted_expected = np.frombuffer(aes.encrypt(initial_np_copy.tobytes()), dtype=np.int8)
                            self.assertTrue(np.array_equal(encrypted_np, encrypted_expected))

                            encrypted_np_copy = np.copy(encrypted_np)

                            csprng.decrypt(initial, initial, key, "aes128", mode)
                            decrypted_np = to_bytes(initial)
                            aes = create_aes(mode, key_np)
                            decrypted_expected = np.frombuffer(aes.decrypt(encrypted_np_copy.tobytes()), dtype=np.int8)
                            self.assertTrue(np.array_equal(decrypted_np, decrypted_expected))

                            self.assertTrue(np.array_equal(initial_np_copy, decrypted_np))

if __name__ == '__main__':
    unittest.main()
