# PyTorch/CSPRNG

CSPRNG is a [PyTorch C++/CUDA extension](https://pytorch.org/tutorials/advanced/cpp_extension.html) that provides [cryptographically secure pseudorandom number generators](https://en.wikipedia.org/wiki/Cryptographically_secure_pseudorandom_number_generator) for PyTorch.

[![CircleCI](https://circleci.com/gh/pytorch/csprng.svg?style=shield&circle-token=64701692dd7f13f31019612289f0200fdb661dc2)](https://circleci.com/gh/pytorch/csprng)

## Featues

CSPRNG exposes four methods to create crypto-secure and non-crypto-secure PRNGs:

| PRNG                                      | Is crypto-secure? | Has seed? |
|-------------------------------------------|-------------------|-----------|
| create_random_device_generator            |         yes       |    no     |
| create_random_device_generator_with_token |         yes       |    no     |
| create_mt19937_generator                  |         no        |    yes    |
| create_mt19937_generator_with_seed        |         no        |    yes    |

The following list of methods supports all forementioned PRNGs:

| Kernel                 | CUDA | CPU |
|------------------------|------|-----|
| random_()              | yes  | no  |
| random_(to)            | yes  | no  |
| random_(from, to)      | yes  | no  |
| uniform_(from, to)     | yes  | no  |
| normal_(mean, std)     | yes  | no  |
| cauchy_(median, sigma) | yes  | no  |
| log_normal_(mean, std) | yes  | no  |
| geometric_(p)          | yes  | no  |
| exponential_(lambda)   | yes  | no  |

## How to build

Since CSPRNG is C++/CUDA extension it uses setuptools, just run `python setup.py install` to build and install it.

## How to use

```python
import torch
import torch_csprng as csprng

# Create crypto-secure PRNG from /dev/urandom:
urandom_gen = csprng.create_random_device_generator_with_token('/dev/urandom')

# Create empty boolean tensor on CUDA and initialize it with random values from urandom_gen:
print(torch.empty(10, dtype=torch.bool, device='cuda').random_(generator=urandom_gen))
tensor([ True, False, False,  True, False, False, False,  True, False, False],
       device='cuda:0')

# Create empty int16 tensor on CUDA and initialize it with random values in range [0, 100) from urandom_gen:
print(torch.empty(10, dtype=torch.int16, device='cuda').random_(100, generator=urandom_gen))
tensor([59, 20, 68, 51, 18, 37,  7, 54, 74, 85], device='cuda:0',
       dtype=torch.int16)

# Create non-crypto-secure MT19937 PRNG:
mt19937_gen = csprng.create_mt19937_generator()

print(torch.empty(10, dtype=torch.int64, device='cuda').random_(torch.iinfo(torch.int64).min, to=None, generator=mt19937_gen))
tensor([-7584783661268263470,  2477984957619728163, -3472586837228887516,
        -5174704429717287072,  4125764479102447192, -4763846282056057972,
         -182922600982469112,  -498242863868415842,   728545841957750221,
         7740902737283645074], device='cuda:0')

# Create crypto-secure PRNG from default random device:
default_device_gen = csprng.create_random_device_generator()

print(torch.randn(10, device='cuda', generator=default_device_gen))
tensor([ 1.2885,  0.3240, -1.1813,  0.8629,  0.5714,  2.3720, -0.5627, -0.5551,
        -0.6304,  0.1090], device='cuda:0')

# Create non-crypto-secure MT19937 PRNG with seed
mt19937_gen = csprng.create_mt19937_generator_with_seed(42)

print(torch.empty(10, device='cuda').geometric_(p=0.2, generator=mt19937_gen))
tensor([ 7.,  1.,  8.,  1., 11.,  3.,  1.,  1.,  5., 10.], device='cuda:0')

print(torch.empty(10, device='cuda').geometric_(p=0.2, generator=mt19937_gen))
tensor([ 1.,  1.,  1.,  6.,  1., 13.,  5.,  1.,  3.,  4.], device='cuda:0')

print(torch.empty(10, device='cuda').geometric_(p=0.2, generator=mt19937_gen))
tensor([14.,  5.,  4.,  5.,  1.,  1.,  8.,  1.,  7., 10.], device='cuda:0')

# Recreate MT19937 PRNG with the same seed
mt19937_gen = csprng.create_mt19937_generator_with_seed(42)

print(torch.empty(10, device='cuda').geometric_(p=0.2, generator=mt19937_gen))
tensor([ 7.,  1.,  8.,  1., 11.,  3.,  1.,  1.,  5., 10.], device='cuda:0')

print(torch.empty(10, device='cuda').geometric_(p=0.2, generator=mt19937_gen))
tensor([ 1.,  1.,  1.,  6.,  1., 13.,  5.,  1.,  3.,  4.], device='cuda:0')

print(torch.empty(10, device='cuda').geometric_(p=0.2, generator=mt19937_gen))
tensor([14.,  5.,  4.,  5.,  1.,  1.,  8.,  1.,  7., 10.], device='cuda:0')

```
