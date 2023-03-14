#!/usr/bin/env python3

import argparse
import torch
import torchcsprng as csprng
import time
from contextlib import ExitStack
from pathlib import Path
from torch.profiler import profile, record_function, tensorboard_trace_handler, ProfilerActivity
from typing import Tuple


def bench_randint(
    size: Tuple[int, ...],
    generator: torch.Generator,
    device: torch.device = torch.device("cpu"),
    n_trials: int = 10,
) -> float:
    with record_function("bench_randint"):
        # Warmup
        args = (-2**63, 2**63-1, size)
        kwargs = {
            "dtype": torch.int64,
            "device": device,
            "generator": generator
        }
        for _ in range(5):
            x = torch.randint(*args, **kwargs)

        t0 = time.time()
        for _ in range(n_trials):
            x = torch.randint(*args, **kwargs)
        t1 = time.time()
        return t1 - t0

def bench_randn(
    size: Tuple[int, ...],
    generator: torch.Generator,
    device: torch.device = torch.device("cpu"),
    n_trials: int = 10,
) -> float:
    with record_function("bench_randn"):
        # Warmup
        args = (size,)
        kwargs = {
            "dtype": torch.float32,
            "device": device,
            "generator": generator
        }
        for _ in range(5):
            x = torch.randn(*args, **kwargs)

        t0 = time.time()
        for _ in range(n_trials):
            x = torch.randn(*args, **kwargs)
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
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Run PyTorch profiler"
    )

    args = parser.parse_args()
    device = torch.device(args.device)

    with ExitStack() as stack:
        if args.profile:
            tracedir = Path.cwd() / "traces"
            tracedir.mkdir(exist_ok=True)
            print("Removing old profiling traces")
            for old_trace in tracedir.glob("*.json.gz"):
                old_trace.unlink()
                print(f"Removed {str(old_trace)}")

            trace_handler = tensorboard_trace_handler(tracedir, use_gzip=True)
            kwargs = {
                "record_shapes": True,
                "with_stack": True,
                "on_trace_ready": trace_handler,
            }
            stack.enter_context(profile(activities=[ProfilerActivity.CPU], **kwargs))

        gen = torch.Generator()
        print(f"Default RNG time:   {bench_randint((args.size,), gen, device):.4f}")
        print(f"Default randn time: {bench_randn((args.size,), gen, device):.4f}")

        gen = csprng.create_generator()
        print(f"CSPRNG time:        {bench_randint((args.size,), gen, device):.4f}")
        print(f"CSPRNG randn time:  {bench_randn((args.size,), gen, device):.4f}")
