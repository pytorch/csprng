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

try:
    import torchcsprng as csprng
except ImportError:
    raise RuntimeError("CSPRNG not available")

class TestCSPRNG(unittest.TestCase):

    all_generators = [
        csprng.create_random_device_generator(),
        csprng.create_random_device_generator('/dev/urandom'),
        csprng.create_mt19937_generator(),
        csprng.create_mt19937_generator(42)
    ]

    int_dtypes = [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]

    fp_ftypes = [torch.float, torch.double]

    num_dtypes = int_dtypes + fp_ftypes

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
                    else:
                        to_inc = torch.iinfo(dtype).max

                    t = torch.empty(self.size, dtype=dtype, device=device).random_(generator=gen)
                    res = stats.kstest(t.cpu(), stats.randint.cdf, args=(0, to_inc))
                    self.assertTrue(res.statistic < 0.1)

    @unittest.skipIf(not torch.cuda.is_available() or not csprng.supports_cuda(), "CUDA is not available or csprng was not compiled with CUDA support")
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
                    res = stats.kstest(t.cpu(), stats.randint.cdf, args=(0, to_))
                    self.assertTrue(res.statistic < 0.1)

    @unittest.skipIf(not torch.cuda.is_available() or not csprng.supports_cuda(), "CUDA is not available or csprng was not compiled with CUDA support")
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
                                res = stats.kstest(t.cpu(), stats.randint.cdf, args=(from_, to_))
                                self.assertTrue(res.statistic < 0.2)

    @unittest.skipIf(not torch.cuda.is_available() or not csprng.supports_cuda(), "CUDA is not available or csprng was not compiled with CUDA support")
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

    @unittest.skipIf(not torch.cuda.is_available() or not csprng.supports_cuda(), "CUDA is not available or csprng was not compiled with CUDA support")
    def test_random_bool_cpu_vs_cuda(self):
        gen = csprng.create_mt19937_generator(42)
        cpu_t = torch.empty(self.size, dtype=torch.bool, device='cpu').random_(generator=gen)
        gen = csprng.create_mt19937_generator(42)
        cuda_t = torch.empty(self.size, dtype=torch.bool, device='cuda').random_(generator=gen)
        self.assertTrue((cpu_t == cuda_t.cpu()).all())

    def test_uniform_kstest(self):
        for device in self.all_devices:
            for gen in self.all_generators:
                for dtype in self.fp_ftypes:
                    for from_ in [-42, 0, 4.2]:
                        for to_ in [-4.2, 0, 42]:
                            if to_ > from_:
                                t = torch.empty(self.size, dtype=dtype, device=device).uniform_(from_, to_, generator=gen)
                                res = stats.kstest(t.cpu().to(torch.double), 'uniform', args=(from_, (to_ - from_)))
                                self.assertTrue(res.statistic < 0.1)

    @unittest.skipIf(not torch.cuda.is_available() or not csprng.supports_cuda(), "CUDA is not available or csprng was not compiled with CUDA support")
    def test_uniform_cpu_vs_cuda(self):
        for dtype in self.fp_ftypes:
            for from_ in [-42, 0, 4.2]:
                for to_ in [-4.2, 0, 42]:
                    if to_ > from_:
                        gen = csprng.create_mt19937_generator(42)
                        cpu_t = torch.empty(self.size, dtype=dtype, device='cpu').uniform_(from_, to_, generator=gen)
                        gen = csprng.create_mt19937_generator(42)
                        cuda_t = torch.empty(self.size, dtype=dtype, device='cuda').uniform_(from_, to_, generator=gen)
                        self.assertTrue((cpu_t - cuda_t.cpu()).abs().max() < 1e-9)

    def test_normal_kstest(self):
        for device in self.all_devices:
            for gen in self.all_generators:
                for dtype in self.fp_ftypes:
                    for mean in [-3, 0, 7]:
                        for std in [1, 5, 7]:
                            t = torch.empty(self.size, dtype=dtype, device=device).normal_(mean=mean, std=std, generator=gen)
                            res = stats.kstest(t.cpu().to(torch.double), 'norm', args=(mean, std))
                            self.assertTrue(res.statistic < 0.1)

    @unittest.skipIf(not torch.cuda.is_available() or not csprng.supports_cuda(), "CUDA is not available or csprng was not compiled with CUDA support")
    def test_normal_cpu_vs_cuda(self):
        for dtype in self.fp_ftypes:
            for mean in [-3, 0, 7]:
                for std in [1, 5, 7]:
                    gen = csprng.create_mt19937_generator(42)
                    cpu_t = torch.empty(self.size, dtype=dtype, device='cpu').normal_(mean=mean, std=std, generator=gen)
                    gen = csprng.create_mt19937_generator(42)
                    cuda_t = torch.empty(self.size, dtype=dtype, device='cuda').normal_(mean=mean, std=std, generator=gen)
                    self.assertTrue((cpu_t - cuda_t.cpu()).abs().max() < 1e-9)

    def test_log_normal_kstest(self):
        for device in self.all_devices:
            for gen in self.all_generators:
                for dtype in self.fp_ftypes:
                    for mean in [-3, 0, 7]:
                        for std in [1, 5, 7]:
                            t = torch.empty(self.size, dtype=dtype, device=device).log_normal_(mean=mean, std=std, generator=gen)
                            res = stats.kstest(t.cpu().to(torch.double), 'lognorm', args=(std, 0, math.exp(mean)))
                            self.assertTrue(res.statistic < 0.1)

    @unittest.skipIf(not torch.cuda.is_available() or not csprng.supports_cuda(), "CUDA is not available or csprng was not compiled with CUDA support")
    def test_log_normal_cpu_vs_cuda(self):
        for dtype in self.fp_ftypes:
            for mean in [-3, 0, 7]:
                for std in [1, 5, 7]:
                    gen = csprng.create_mt19937_generator(42)
                    cpu_t = torch.empty(self.size, dtype=dtype, device='cpu').log_normal_(mean=mean, std=std, generator=gen)
                    gen = csprng.create_mt19937_generator(42)
                    cuda_t = torch.empty(self.size, dtype=dtype, device='cuda').log_normal_(mean=mean, std=std, generator=gen)
                    self.assertTrue((cpu_t - cuda_t.cpu()).abs().max() < 1e-4)

    def test_exponential_kstest(self):
        for device in self.all_devices:
            for gen in self.all_generators:
                for dtype in self.fp_ftypes:
                    for lambd in [0.5, 1.0, 5.0]:
                        t = torch.empty(self.size, dtype=dtype, device=device).exponential_(lambd=lambd, generator=gen)
                        res = stats.kstest(t.cpu().to(torch.double), 'expon', args=(0, 1 / lambd,))
                        self.assertTrue(res.statistic < 0.1)

    @unittest.skipIf(not torch.cuda.is_available() or not csprng.supports_cuda(), "CUDA is not available or csprng was not compiled with CUDA support")
    def test_exponential_cpu_vs_cuda(self):
        for dtype in self.fp_ftypes:
            for lambd in [0.5, 1.0, 5.0]:
                gen = csprng.create_mt19937_generator(42)
                cpu_t = torch.empty(self.size, dtype=dtype, device='cpu').exponential_(lambd=lambd, generator=gen)
                gen = csprng.create_mt19937_generator(42)
                cuda_t = torch.empty(self.size, dtype=dtype, device='cuda').exponential_(lambd=lambd, generator=gen)
                self.assertTrue((cpu_t - cuda_t.cpu()).abs().max() < 1e-9)

    def test_cauchy_kstest(self):
        for device in self.all_devices:
            for gen in self.all_generators:
                for dtype in self.fp_ftypes:
                    for median in [-10, 0, 50]:
                        for sigma in [0.5, 1.0, 10.0]:
                            t = torch.empty(self.size, dtype=dtype, device=device).cauchy_(median=median, sigma=sigma, generator=gen)
                            res = stats.kstest(t.cpu().to(torch.double), 'cauchy', args=(median, sigma))
                            self.assertTrue(res.statistic < 0.1)

    @unittest.skipIf(not torch.cuda.is_available() or not csprng.supports_cuda(), "CUDA is not available or csprng was not compiled with CUDA support")
    def test_cauchy_cpu_vs_cuda(self):
        for dtype in self.fp_ftypes:
            for median in [-10, 0, 50]:
                for sigma in [0.5, 1.0, 10.0]:
                    gen = csprng.create_mt19937_generator(42)
                    cpu_t = torch.empty(self.size, dtype=dtype, device='cpu').cauchy_(median=median, sigma=sigma, generator=gen)
                    gen = csprng.create_mt19937_generator(42)
                    cuda_t = torch.empty(self.size, dtype=dtype, device='cuda').cauchy_(median=median, sigma=sigma, generator=gen)
                    self.assertTrue((cpu_t - cuda_t.cpu()).abs().max() < 1e-9)

    def test_geometric(self):
        for device in self.all_devices:
            for gen in self.all_generators:
                for dtype in self.fp_ftypes:
                    for p in [0.2, 0.5, 0.8]:
                        t = torch.empty(self.size, dtype=dtype, device=device).geometric_(p=p, generator=gen)
                        # actual = np.histogram(t.cpu().to(torch.double), np.arange(1, 100))[0]
                        # expected = stats.geom(p).pmf(np.arange(1, 99)) * self.size
                        # res = stats.chisquare(actual, expected)
                        # self.assertAlmostEqual(res.pvalue, 1.0, delta=0.5) TODO https://github.com/pytorch/csprng/issues/7

    @unittest.skipIf(not torch.cuda.is_available() or not csprng.supports_cuda(), "CUDA is not available or csprng was not compiled with CUDA support")
    def test_geometric_cpu_vs_cuda(self):
        for dtype in self.fp_ftypes:
            for p in [0.2, 0.5, 0.8]:
                gen = csprng.create_mt19937_generator(42)
                cpu_t = torch.empty(self.size, dtype=dtype, device='cpu').geometric_(p=p, generator=gen)
                gen = csprng.create_mt19937_generator(42)
                cuda_t = torch.empty(self.size, dtype=dtype, device='cuda').geometric_(p=p, generator=gen)
                self.assertTrue((cpu_t - cuda_t.cpu()).abs().max() < 1e-9)

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

    @unittest.skipIf(torch.get_num_threads() < 2, "requires multithreading CPU")
    def test_cpu_parallel(self):
        urandom_gen = csprng.create_random_device_generator('/dev/urandom')

        def measure(size):
            t = torch.empty(size, dtype=torch.float32, device='cpu')
            start = time.time()
            for i in range(10):
                t.normal_(generator=urandom_gen)
            finish = time.time()
            return finish - start

        time_for_1K = measure(1000)
        time_for_1M = measure(1000000)
        # Pessimistic check that parallel execution gives >= 1.5 performance boost
        self.assertTrue(time_for_1M/time_for_1K < 1000 / min(1.5, torch.get_num_threads()))

    @unittest.skip("Temporary disable because doesn't work on Sandcastle")
    def test_version(self):
        import torchcsprng.version as version
        self.assertTrue(version.__version__)
        self.assertTrue(version.git_version)

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

    def test_aes128_key_tensor(self):
        size = 10
        for gen in self.all_generators:
            s = set()
            for _ in range(0, size):
                t = csprng.aes128_key_tensor(gen)
                s.add(str(t))
            self.assertEqual(len(s), size)

    def test_const_generator(self):
        for device in self.all_devices:
            for gen in self.all_generators:
                for dtype in self.int_dtypes:
                    key = csprng.aes128_key_tensor(gen)
                    const_gen = csprng.create_const_generator(key)
                    first = torch.empty(self.size, dtype=dtype, device=device).random_(generator=const_gen)
                    second = torch.empty(self.size, dtype=dtype, device=device).random_(generator=const_gen)
                    self.assertTrue((first - second).max().abs() == 0)

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

        for device in self.all_devices:
            for key_dtype in self.all_dtypes:
                key_size = key_size_bytes // sizeof(key_dtype)
                key = torch.empty(key_size, dtype=key_dtype, device=device).random_()
                for initial_dtype in self.all_dtypes:
                    for encrypted_dtype in self.all_dtypes:
                        for decrypted_dtype in self.all_dtypes:
                            for initial_size in [0, 4, 8, 15, 16, 23, 42]:
                                for mode in ["ecb", "ctr"]:
                                    encrypted_size = (initial_size * sizeof(initial_dtype) + block_size_bytes - 1) // block_size_bytes * block_size_bytes // sizeof(encrypted_dtype)
                                    decrypted_size = (encrypted_size * sizeof(encrypted_dtype) + block_size_bytes - 1) // block_size_bytes * block_size_bytes // sizeof(decrypted_dtype)

                                    initial = torch.empty(initial_size, dtype=initial_dtype, device=device).random_()
                                    encrypted = torch.empty(encrypted_size, dtype=encrypted_dtype, device=device).random_()
                                    decrypted = torch.empty(decrypted_size, dtype=decrypted_dtype, device=device).random_()

                                    initial_np = initial.cpu().numpy().view(np.int8)
                                    decrypted_np = decrypted.cpu().numpy().view(np.int8)
                                    padding_size_bytes = initial_size * sizeof(initial_dtype) - decrypted_size * sizeof(decrypted_dtype)
                                    if padding_size_bytes != 0:
                                        decrypted_np = decrypted_np[:padding_size_bytes]

                                    csprng.encrypt(initial, encrypted, key, "aes128", mode)

                                    if initial_size > 8:
                                        self.assertFalse(np.array_equal(initial_np, decrypted_np))

                                    csprng.decrypt(encrypted, decrypted, key, "aes128", mode)
                                    decrypted_np = decrypted.cpu().numpy().view(np.int8)
                                    if padding_size_bytes != 0:
                                        decrypted_np = decrypted_np[:padding_size_bytes]

                                    self.assertTrue(np.array_equal(initial_np, decrypted_np))

if __name__ == '__main__':
    unittest.main()
