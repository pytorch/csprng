# PyTorch/CSPRNG

[![CircleCI](https://circleci.com/gh/pytorch/csprng.svg?style=shield&circle-token=64701692dd7f13f31019612289f0200fdb661dc2)](https://circleci.com/gh/pytorch/csprng)

torchcsprng is a [PyTorch C++/CUDA extension](https://pytorch.org/tutorials/advanced/cpp_extension.html) that provides:

- [AES](https://en.wikipedia.org/wiki/Advanced_Encryption_Standard) 128-bit encryption/decryption in two modes: [ECB](https://en.wikipedia.org/wiki/Block_cipher_mode_of_operation#Electronic_codebook_(ECB)) and [CTR](https://en.wikipedia.org/wiki/Block_cipher_mode_of_operation#Counter_(CTR)) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pytorch/csprng/blob/master/examples/encrypt_decrypt.ipynb)
- [cryptographically secure pseudorandom number generators](https://en.wikipedia.org/wiki/Cryptographically_secure_pseudorandom_number_generator) for PyTorch. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pytorch/csprng/blob/master/examples/csprng.ipynb)

## Design

torchcsprng generates a random 128-bit key on CPU using one of its generators and runs
[AES128](https://en.wikipedia.org/wiki/Advanced_Encryption_Standard) in [CTR mode](https://en.wikipedia.org/wiki/Block_cipher_mode_of_operation#Counter_(CTR))
 either on CPU or on GPU using CUDA to generate a random 128 bit state and apply a transformation function to map it to target tensor values.
This approach is based on [Parallel Random Numbers: As Easy as 1, 2, 3(John K. Salmon, Mark A. Moraes, Ron O. Dror, and David E. Shaw, D. E. Shaw Research)](http://www.thesalmons.org/john/random123/papers/random123sc11.pdf).
It makes torchcsprng both crypto-secure and parallel on CUDA and CPU.

![CSPRNG architecture](.github/csprng_architecture.png)

Advantages:

- The user can choose either seed-based(for testing) or random device based(fully crypto-secure) generators
- One generator instance for both CPU and CUDA tensors(because the encryption key is always generated on CPU)
- CPU random number generation is also parallel(unlike the default PyTorch CPU generator)

## Features

torchcsprng 0.2.0 exposes new API for tensor encryption/decryption. Tensor encryption/decryption API is dtype agnostic, so a tensor of any dtype can be encrypted and the result can be stored to a tensor of any dtype. An encryption key also can be a tensor of any dtype. Currently torchcsprng supports [AES](https://en.wikipedia.org/wiki/Advanced_Encryption_Standard) cipher with 128-bit key in two modes: [ECB](https://en.wikipedia.org/wiki/Block_cipher_mode_of_operation#Electronic_codebook_(ECB)) and [CTR](https://en.wikipedia.org/wiki/Block_cipher_mode_of_operation#Counter_(CTR)).

* `torchcsprng.encrypt(input: Tensor, output: Tensor, key: Tensor, cipher: string, mode: string)`

> - `input` tensor can be any CPU or CUDA tensor of any dtype and size in bytes(zero-padding is used to make its size in bytes divisible by block size in bytes)
> - `output` tensor can have any dtype and the same device as `input` tensor and the size in bytes rounded up to the block size in bytes(16 bytes for AES 128)
> - `key` tensor can have any dtype and the same device as `input` tensor and size in bytes equal to 16 for AES 128
> - `cipher` currently can be only one supported value `"aes128"`
> - `mode` currently can be either [`"ecb"`](https://en.wikipedia.org/wiki/Block_cipher_mode_of_operation#Electronic_codebook_(ECB)) or [`"ctr"`](https://en.wikipedia.org/wiki/Block_cipher_mode_of_operation#Counter_(CTR))

* `torchcsprng.decrypt(input: Tensor, output: Tensor, key: Tensor, cipher: string, mode: string)`

> - `input` tensor can be any CPU or CUDA tensor of any dtype with size in bytes divisible by the block size in bytes(16 bytes for AES 128)
> - `output` tensor can have any dtype but the same device as `input` tensor and the same size in bytes as `input` tensor
> - `key` tensor can have any dtype and the same device as `input` tensor and size in bytes equal to 16 for AES 128
> - `cipher` currently can be only one supported value `"aes128"`
> - `mode` currently can be either [`"ecb"`](https://en.wikipedia.org/wiki/Block_cipher_mode_of_operation#Electronic_codebook_(ECB)) or [`"ctr"`](https://en.wikipedia.org/wiki/Block_cipher_mode_of_operation#Counter_(CTR))

torchcsprng exposes two methods to create crypto-secure and non-crypto-secure PRNGs:

| Method to create PRNG                              | Is crypto-secure? | Has seed? | Underlying implementation |
|----------------------------------------------------|-------------------|-----------|---------------------------|
| create_random_device_generator(token: string=None) |         yes       |    no     | See [std::random_device](https://en.cppreference.com/w/cpp/numeric/random/random_device) and [its constructor](https://en.cppreference.com/w/cpp/numeric/random/random_device/random_device). The implementation in libstdc++ expects token to name the source of random bytes. Possible token values include "default", "rand_s", "rdseed", "rdrand", "rdrnd", "/dev/urandom", "/dev/random", "mt19937", and integer string specifying the seed of the mt19937 engine. (Token values other than "default" are only valid for certain targets.) If token=None then constructs a new std::random_device object with an implementation-defined token. |
| create_mt19937_generator(seed: int=None)           |         no        |    yes    | See [std::mt19937](https://en.cppreference.com/w/cpp/numeric/random/mersenne_twister_engine) and [its constructor](https://en.cppreference.com/w/cpp/numeric/random/mersenne_twister_engine/mersenne_twister_engine). Constructs a mersenne_twister_engine object, and initializes its internal state sequence to pseudo-random values. If seed=None then seeds the engine with default_seed.|

The following list of methods supports all forementioned PRNGs:

| Kernel                 | CUDA | CPU |
|------------------------|------|-----|
| random_()              | yes  | yes |
| random_(to)            | yes  | yes |
| random_(from, to)      | yes  | yes |
| uniform_(from, to)     | yes  | yes |
| normal_(mean, std)     | yes  | yes |
| cauchy_(median, sigma) | yes  | yes |
| log_normal_(mean, std) | yes  | yes |
| geometric_(p)          | yes  | yes |
| exponential_(lambda)   | yes  | yes |
| randperm(n)            | yes* | yes |

* the calculations are done on CPU and the result is copied to CUDA

## Installation

CSPRNG works with Python 3.6-3.9 on the following operating systems and can be used with PyTorch tensors on the following devices:

| Tensor Device Type | Linux     | macOS         | MS Window      |
|--------------------|-----------|---------------|----------------| 
| CPU                | Supported | Supported     | Supported      |
| CUDA               | Supported | Not Supported | Supported since 0.2.0 |

The following is the corresponding CSPRNG versions and supported Python versions.

| PyTorch | CSPRNG | Python   | CUDA             |
|---------|--------|----------|------------------|
| 1.8.0   | 0.2.0  | 3.7-3.9  | 10.1, 10.2, 11.1 |
| 1.7.1   | 0.1.4  | 3.6-3.8  | 9.2, 10.1, 10.2  |
| 1.7.0   | 0.1.3  | 3.6-3.8  | 9.2, 10.1, 10.2  |
| 1.6.0   | 0.1.2  | 3.6-3.8  | 9.2, 10.1, 10.2  |


### Binary Installation

Anaconda:

| OS            | CUDA                                           |                                                                                                                                                                                                                                                                                                       |
|---------------|------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Linux/Windows | 10.1<br/><br/>10.2<br/><br/>11.1<br/><br/>None | conda install torchcsprng cudatoolkit=10.1 -c pytorch -c conda-forge<br/><br/>conda install torchcsprng cudatoolkit=10.2 -c pytorch -c conda-forge<br/><br/>conda install torchcsprng cudatoolkit=11.1 -c pytorch -c conda-forge<br/><br/>conda install torchcsprng cpuonly -c pytorch -c conda-forge |
| macOS         | None                                           | conda install torchcsprng -c pytorch                                                                                                                                                                                                                                                          |

pip:

| OS            | CUDA                                           |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
|---------------|------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Linux/Windows | 10.1<br/><br/>10.2<br/><br/>11.1<br/><br/>None | pip install torchcsprng==0.2.0+cu101 torch==1.8.0+cu101 -f https://download.pytorch.org/whl/cu101/torch_stable.html <br/><br/>pip install torchcsprng==0.2.0 torch==1.8.0 -f https://download.pytorch.org/whl/cu102/torch_stable.html <br/><br/>pip install torchcsprng==0.2.0+cu111 torch==1.8.0+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html <br/><br/>pip install torchcsprng==0.2.0+cpu torch==1.8.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html |
| macOS         | None                                           | pip install torchcsprng torch                                                                                                                                                                                                                                                                                                                                                                                                                                                     |

### Nightly builds:

Anaconda:

| OS            | CUDA                                           |                                                                                                                                                                                                                                                                                                                                       |
|---------------|------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Linux/Windows | 10.1<br/><br/>10.2<br/><br/>11.1<br/><br/>None | conda install torchcsprng cudatoolkit=10.1 -c pytorch-nightly -c conda-forge<br/><br/>conda install torchcsprng cudatoolkit=10.2 -c pytorch-nightly -c conda-forge<br/><br/>conda install torchcsprng cudatoolkit=11.1 -c pytorch-nightly -c conda-forge<br/><br/>conda install torchcsprng cpuonly -c pytorch-nightly -c conda-forge |
| macOS         | None                                           | conda install torchcsprng -c pytorch-nightly                                                                                                                                                                                                                                                                                          |

pip:

| OS            | CUDA                                           |                                                                                                                                                                                                                                                                                                                                                                                                                                            |
|---------------|------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Linux/Windows | 10.1<br/><br/>10.2<br/><br/>11.1<br/><br/>None | pip install --pre torchcsprng -f https://download.pytorch.org/whl/nightly/cu101/torch_nightly.html <br/><br/> pip install --pre torchcsprng -f https://download.pytorch.org/whl/nightly/cu102/torch_nightly.html <br/><br/> pip install --pre torchcsprng -f https://download.pytorch.org/whl/nightly/cu111/torch_nightly.html <br/><br/> pip install --pre torchcsprng -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html |
| macOS         | None                                           | pip install --pre torchcsprng -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html                                                                                                                                                                                                                                                                                                                                           |

### From Source

torchcsprng is a Python C++/CUDA extension that depends on PyTorch. In order to build CSPRNG from source it is required to have Python(>=3.7) with PyTorch(>=1.8.0) installed and C++ compiler(gcc/clang for Linux, XCode for macOS, Visual Studio for MS Windows).
To build torchcsprng you can run the following:
```console
python setup.py install
```
By default, GPU support is built if CUDA is found and torch.cuda.is_available() is True. Additionally, it is possible to force building GPU support by setting the FORCE_CUDA=1 environment variable, which is useful when building a docker image.

## Getting Started

The torchcsprng API is available in `torchcsprng` module:
```python
import torch
import torchcsprng as csprng
```
Create crypto-secure PRNG from /dev/urandom:
```python
urandom_gen = csprng.create_random_device_generator('/dev/urandom')
```

Create empty boolean tensor on CUDA and initialize it with random values from urandom_gen:
```python
torch.empty(10, dtype=torch.bool, device='cuda').random_(generator=urandom_gen)
```
```
tensor([ True, False, False,  True, False, False, False,  True, False, False],
       device='cuda:0')
```

Create empty int16 tensor on CUDA and initialize it with random values in range [0, 100) from urandom_gen:
```python
torch.empty(10, dtype=torch.int16, device='cuda').random_(100, generator=urandom_gen)
```
```
tensor([59, 20, 68, 51, 18, 37,  7, 54, 74, 85], device='cuda:0',
       dtype=torch.int16)
```

Create non-crypto-secure MT19937 PRNG:
```python
mt19937_gen = csprng.create_mt19937_generator()
torch.empty(10, dtype=torch.int64, device='cuda').random_(torch.iinfo(torch.int64).min, to=None, generator=mt19937_gen)
```
```
tensor([-7584783661268263470,  2477984957619728163, -3472586837228887516,
        -5174704429717287072,  4125764479102447192, -4763846282056057972,
         -182922600982469112,  -498242863868415842,   728545841957750221,
         7740902737283645074], device='cuda:0')
```

Create crypto-secure PRNG from default random device:
```python
default_device_gen = csprng.create_random_device_generator()
torch.randn(10, device='cuda', generator=default_device_gen)
```
```
tensor([ 1.2885,  0.3240, -1.1813,  0.8629,  0.5714,  2.3720, -0.5627, -0.5551,
        -0.6304,  0.1090], device='cuda:0')
```

Create non-crypto-secure MT19937 PRNG with seed:
```python
mt19937_gen = csprng.create_mt19937_generator(42)
torch.empty(10, device='cuda').geometric_(p=0.2, generator=mt19937_gen)
```
```
tensor([ 7.,  1.,  8.,  1., 11.,  3.,  1.,  1.,  5., 10.], device='cuda:0')
```

Recreate MT19937 PRNG with the same seed:
```python
mt19937_gen = csprng.create_mt19937_generator(42)
torch.empty(10, device='cuda').geometric_(p=0.2, generator=mt19937_gen)
```
```
tensor([ 7.,  1.,  8.,  1., 11.,  3.,  1.,  1.,  5., 10.], device='cuda:0')
```

## Contributing
We appreciate all contributions. If you are planning to contribute back bug-fixes, please do so without any further discussion. If you plan to contribute new features, utility functions or extensions, please first open an issue and discuss the feature with us.



## License

torchcsprng is BSD 3-clause licensed. See the license file [here](https://github.com/pytorch/csprng/blob/master/LICENSE)

## [Terms of Use](https://opensource.facebook.com/legal/terms)

## [Privacy Policy](https://opensource.facebook.com/legal/privacy)
