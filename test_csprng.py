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

    all_dtypes = num_dtypes + [torch.bool]

    size = 1000

    device = 'cuda'

    def test_random(self):
        for gen in self.all_generators:
            for dtype in self.num_dtypes:
                if dtype == torch.float:
                    to_inc = 2**24
                elif dtype == torch.double:
                    to_inc = 2**53
                else:
                    to_inc = torch.iinfo(dtype).max

                t = torch.empty(self.size, dtype=dtype, device=self.device).random_(generator=gen)
                res = stats.kstest(t.cpu(), stats.randint.cdf, args=(0, to_inc))
                self.assertTrue(res.statistic < 0.1)

    def test_random_to(self):
        to_ = 42
        for gen in self.all_generators:
            for dtype in self.num_dtypes:
                t = torch.zeros(self.size, dtype=dtype, device=self.device).random_(to_, generator=gen)
                res = stats.kstest(t.cpu(), stats.randint.cdf, args=(0, to_))
                self.assertTrue(res.statistic < 0.1)

    def test_random_from_to(self):
        for gen in self.all_generators:
            for dtype in self.num_dtypes:
                for from_ in [0, 24, 42]:
                    for to_ in [42, 99, 123]:
                        if from_ < to_:
                            t = torch.zeros(self.size, dtype=dtype, device=self.device).random_(from_, to_, generator=gen)
                            res = stats.kstest(t.cpu(), stats.randint.cdf, args=(from_, to_))
                            self.assertTrue(res.statistic < 0.2)

    def test_random_bool(self):
        for gen in self.all_generators:
            t = torch.empty(self.size, dtype=torch.bool, device=self.device)

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

    def test_uniform(self):
        for gen in self.all_generators:
            for dtype in self.fp_ftypes:
                for from_ in [-42, 0, 4.2]:
                    for to_ in [-4.2, 0, 42]:
                        if to_ > from_:
                            t = torch.empty(self.size, dtype=dtype, device=self.device).uniform_(from_, to_, generator=gen)
                            res = stats.kstest(t.cpu().to(torch.double), 'uniform', args=(from_, (to_ - from_)))
                            self.assertTrue(res.statistic < 0.1)

    def test_normal(self):
        for gen in self.all_generators:
            for dtype in self.fp_ftypes:
                for mean in [-3, 0, 7]:
                    for std in [1, 5, 7]:
                        t = torch.empty(self.size, dtype=dtype, device=self.device).normal_(mean=mean, std=std, generator=gen)
                        res = stats.kstest(t.cpu().to(torch.double), 'norm', args=(mean, std))
                        self.assertTrue(res.statistic < 0.1)

    def test_log_normal(self):
        for gen in self.all_generators:
            for dtype in self.fp_ftypes:
                for mean in [-3, 0, 7]:
                    for std in [1, 5, 7]:
                        t = torch.empty(self.size, dtype=dtype, device=self.device).log_normal_(mean=mean, std=std, generator=gen)
                        res = stats.kstest(t.cpu().to(torch.double), 'lognorm', args=(std, 0, math.exp(mean)))
                        self.assertTrue(res.statistic < 0.1)

    def test_exponential(self):
        for gen in self.all_generators:
            for dtype in self.fp_ftypes:
                for lambd in [0.5, 1.0, 5.0]:
                    t = torch.empty(self.size, dtype=dtype, device=self.device).exponential_(lambd=lambd, generator=gen)
                    res = stats.kstest(t.cpu().to(torch.double), 'expon', args=(0, 1 / lambd,))
                    self.assertTrue(res.statistic < 0.1)

    def test_cauchy(self):
        for gen in self.all_generators:
            for dtype in self.fp_ftypes:
                for median in [-10, 0, 50]:
                    for sigma in [0.5, 1.0, 10.0]:
                        t = torch.empty(self.size, dtype=dtype, device=self.device).cauchy_(median=median, sigma=sigma, generator=gen)
                        res = stats.kstest(t.cpu().to(torch.double), 'cauchy', args=(median, sigma))
                        self.assertTrue(res.statistic < 0.1)

    def test_geometric(self):
        for gen in self.all_generators:
            for dtype in self.fp_ftypes:
                for p in [0.2, 0.5, 0.8]:
                    t = torch.empty(self.size, dtype=dtype, device=self.device).geometric_(p=p, generator=gen)
                    actual = np.histogram(t.cpu().to(torch.double), np.arange(1, 100))[0]
                    expected = stats.geom(p).pmf(np.arange(1, 99)) * self.size
                    res = stats.chisquare(actual, expected)
                    self.assertAlmostEqual(res.pvalue, 1.0, delta=0.1)

if __name__ == '__main__':
    unittest.main()
