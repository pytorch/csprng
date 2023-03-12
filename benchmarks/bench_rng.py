#!/usr/bin/env python3

import argparse
import torch
import torchcsprng as csprng
import time
from typing import Tuple


def bench_randint(
    size: Tuple[int, ...],
    generator: torch.Generator,
    device: torch.device = torch.device("cpu"),
    n_trials: int = 10,
) -> float:
    # Warmup
    args = (-2**63, 2**63-1, size)
    kwargs = {
        "dtype": torch.int64,
        "device": device,
        "generator": generator
    }
    _ = torch.randint(*args, **kwargs)

    t0 = time.time()
    for _ in range(n_trials):
        x = torch.randint(*args, **kwargs)
    t1 = time.time()
    return t1 - t0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="bench_rng",
        description="Benchmarking for PyTorch CSPRNG",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Select device to perform benchmarking",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=2**10,
        help="Number of random elements to generate",
    )

    args = parser.parse_args()
    device = torch.device(args.device)

    generator = torch.Generator()
    time_default = bench_randint((args.size,), generator, device)
    print(f"Default RNG time: {time_default:.4f}")

    generator = csprng.create_random_device_generator()
    time_csprng = bench_randint((args.size,), generator, device)
    print(f"CSPRNG time: {time_csprng:.4f}")
