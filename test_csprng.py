import unittest
import torch
from scipy import stats
import numpy as np
import math

try:
    import torch_csprng as csprng
except ImportError:
    raise RuntimeError("CSPRNG not available")

class TestCSPRNG(unittest.TestCase):

    all_generators = [
        csprng.create_random_device_generator(),
        csprng.create_random_device_generator_with_token('/dev/urandom'),
        csprng.create_mt19937_generator(),
        csprng.create_mt19937_generator_with_seed(42)
    ]

    int_dtypes = [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]

    fp_ftypes = [torch.float, torch.double]

    num_dtypes = int_dtypes + fp_ftypes

    size = 1000

    all_devices = ['cpu', 'cuda']

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

    def test_random_cpu_vs_cuda(self):
        for dtype in self.num_dtypes:
            gen = csprng.create_mt19937_generator_with_seed(42)
            cpu_t = torch.empty(self.size, dtype=dtype, device='cpu').random_(generator=gen)
            gen = csprng.create_mt19937_generator_with_seed(42)
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

    def test_random_to_cpu_vs_cuda(self):
        to_ = 42
        for dtype in self.num_dtypes:
            gen = csprng.create_mt19937_generator_with_seed(42)
            cpu_t = torch.zeros(self.size, dtype=dtype, device='cpu').random_(to_, generator=gen)
            gen = csprng.create_mt19937_generator_with_seed(42)
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

    def test_random_from_to_cpu_vs_cuda(self):
        for dtype in self.num_dtypes:
            for from_ in [0, 24, 42]:
                for to_ in [42, 99, 123]:
                    if from_ < to_:
                        gen = csprng.create_mt19937_generator_with_seed(42)
                        cpu_t = torch.zeros(self.size, dtype=dtype, device='cpu').random_(from_, to_, generator=gen)
                        gen = csprng.create_mt19937_generator_with_seed(42)
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

    def test_random_bool_cpu_vs_cuda(self):
        gen = csprng.create_mt19937_generator_with_seed(42)
        cpu_t = torch.empty(self.size, dtype=torch.bool, device='cpu').random_(generator=gen)
        gen = csprng.create_mt19937_generator_with_seed(42)
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

    def test_uniform_cpu_vs_cuda(self):
        for dtype in self.fp_ftypes:
            for from_ in [-42, 0, 4.2]:
                for to_ in [-4.2, 0, 42]:
                    if to_ > from_:
                        gen = csprng.create_mt19937_generator_with_seed(42)
                        cpu_t = torch.empty(self.size, dtype=dtype, device='cpu').uniform_(from_, to_, generator=gen)
                        gen = csprng.create_mt19937_generator_with_seed(42)
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

    def test_normal_cpu_vs_cuda(self):
        for dtype in self.fp_ftypes:
            for mean in [-3, 0, 7]:
                for std in [1, 5, 7]:
                    gen = csprng.create_mt19937_generator_with_seed(42)
                    cpu_t = torch.empty(self.size, dtype=dtype, device='cpu').normal_(mean=mean, std=std, generator=gen)
                    gen = csprng.create_mt19937_generator_with_seed(42)
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

    def test_log_normal_cpu_vs_cuda(self):
        for dtype in self.fp_ftypes:
            for mean in [-3, 0, 7]:
                for std in [1, 5, 7]:
                    gen = csprng.create_mt19937_generator_with_seed(42)
                    cpu_t = torch.empty(self.size, dtype=dtype, device='cpu').log_normal_(mean=mean, std=std, generator=gen)
                    gen = csprng.create_mt19937_generator_with_seed(42)
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

    def test_exponential_cpu_vs_cuda(self):
        for dtype in self.fp_ftypes:
            for lambd in [0.5, 1.0, 5.0]:
                gen = csprng.create_mt19937_generator_with_seed(42)
                cpu_t = torch.empty(self.size, dtype=dtype, device='cpu').exponential_(lambd=lambd, generator=gen)
                gen = csprng.create_mt19937_generator_with_seed(42)
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

    def test_cauchy_cpu_vs_cuda(self):
        for dtype in self.fp_ftypes:
            for median in [-10, 0, 50]:
                for sigma in [0.5, 1.0, 10.0]:
                    gen = csprng.create_mt19937_generator_with_seed(42)
                    cpu_t = torch.empty(self.size, dtype=dtype, device='cpu').cauchy_(median=median, sigma=sigma, generator=gen)
                    gen = csprng.create_mt19937_generator_with_seed(42)
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

    def test_geometric_cpu_vs_cuda(self):
        for dtype in self.fp_ftypes:
            for p in [0.2, 0.5, 0.8]:
                gen = csprng.create_mt19937_generator_with_seed(42)
                cpu_t = torch.empty(self.size, dtype=dtype, device='cpu').geometric_(p=p, generator=gen)
                gen = csprng.create_mt19937_generator_with_seed(42)
                cuda_t = torch.empty(self.size, dtype=dtype, device='cuda').geometric_(p=p, generator=gen)
                self.assertTrue((cpu_t - cuda_t.cpu()).abs().max() < 1e-9)

if __name__ == '__main__':
    unittest.main()
