import pyrootutils

root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)


import argparse
import os
import time

# from functools import partialmethod
from pathlib import Path

import numpy as np
import torch as T

from src.models.diffusion import euler_maruyama_sampler
from src.models.pc_jedi import TransformerDiffusionGenerator
# from tqdm import tqdm


# tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)


class objectview:
    """converts a dict into an object."""

    def __init__(self, d):
        self.__dict__ = d


def full_gen_time(model, noise, n=10) -> float:
    t = []
    for _ in range(n + 1):
        ts = time.process_time()
        euler_maruyama_sampler(model, model.diff_sched, noise, 200)
        te = time.process_time()
        t.append(te - ts)
    return np.mean(t[1:]) * 1000, np.std(t[1:]) * 1000


def foward_time(model, noise, labels=None, n=10, mode="jetdiff") -> float:
    t = []
    mask = T.full(noise.shape[:-1], True, device=noise.device)
    diff_time = T.rand(len(noise), device=noise.device)
    for _ in range(n + 1):
        ts = time.process_time()
        if mode == "jetdiff":
            model.forward(noise, diff_time, mask, 100)
        elif mode == "mpgan":
            model.forward(noise, labels)
        else:
            raise ValueError(f"Undefined mode '{mode}'")
        te = time.process_time()
        t.append(te - ts)
    return np.mean(t[1:]) * 1000, np.std(t[1:]) * 1000


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--jetdiff-model-dir",
        type=str,
        default="/srv/beegfs/scratch/users/s/senguptd/jet_diffusion/epic_jedi_30",
        help="Path to directory where the model and its checkpoints are saved.",
    )
    parser.add_argument(
        "--jetdiff-model-name",
        type=str,
        default="2023-05-19_10-03-09-602489",
        help="Name of the model to load for generation.",
    )
    parser.add_argument(
        "--mpgan-state-dict",
        type=str,
        default=(
            "/mnt/scratch/git/gitlab/cern/jetdiffusion/mpgan/neurips21/"
            "trained_models/mp_g/G_best_epoch.pt"
        ),
        help="Path to generator's state dict.",
    )
    parser.add_argument(
        "--mpgan-args",
        type=str,
        default=(
            "/mnt/scratch/git/gitlab/cern/jetdiffusion/mpgan/neurips21/"
            "trained_models/mp_g/args.txt"
        ),
        help="Path to generator's args file.",
    )
    parser.add_argument(
        "--mpgan-datasets-path",
        type=str,
        default="/mnt/scratch/git/gitlab/cern/jetdiffusion/mpgan/neurips21/datasets/",
        help="Path to gen jets output file.",
    )

    args = parser.parse_args()

    return args


def find_best_checkpoint(model_path: str) -> str:
    return str(
        sorted(Path(model_path, "checkpoints").glob("*.ckpt"), key=os.path.getmtime)[-1]
    )


def main() -> None:
    # Load the network checkpoint
    args = get_args()

    # Number of averaged tests
    n = 10



    # Jet-Diffusion ###########################################################
    # os.chdir("/mnt/scratch/git/gitlab/cern/jetdiffusion")



    # Load the full model configuration
    full_path = Path(args.jetdiff_model_dir, args.jetdiff_model_name)

    # Load the checkpoint
    ckpt_path = find_best_checkpoint(full_path)
    model = TransformerDiffusionGenerator.load_from_checkpoint(ckpt_path)

    print(".:JeDi (EM-200):.")

    # Create the dummy input and calculate
    noise = T.randn((1, 30, 3))
    mean, std = foward_time(model, noise, n=n)
    print(f"CPU: Forward pass {mean:33.2f} +- {std:.2f} ms")
    mean, std = full_gen_time(model, noise, n=n)
    print(f"CPU: Full Generate (50 steps) {mean:20.2f} +- {std:.2f} ms")

    # Move the the test to the gpu
    model = model.to("cuda")
    noise = noise.to("cuda")
    mean, std = foward_time(model, noise, n=n)
    print(f"GPU: Forward pass {mean:33.2f} +- {std:.2f} ms")
    mean, std = full_gen_time(model, noise, n=n)
    print(f"GPU: Full Generate (50 steps) {mean:20.2f} +- {std:.2f} ms")

    # Perform a batch test of 10 samples
    noise = T.randn((10, 30, 3)).to("cuda")
    mean, std = foward_time(model, noise, n=n)
    print(f"GPU: 10 batch, Forward pass {mean:23.2f} +- {std:.2f} ms")
    mean, std = full_gen_time(model, noise, n=n)
    print(f"GPU: 10 batch, Full Generate (50 steps) {mean:10.2f} +- {std:.2f} ms")

    # Perform a batch test of 1000 samples
    noise = T.randn((1000, 30, 3)).to("cuda")
    mean, std = foward_time(model, noise, n=n)
    print(f"GPU: 1000 batch, Forward pass {mean:21.2f} +- {std:.2f} ms")
    mean, std = full_gen_time(model, noise, n=n)
    print(f"GPU: 1000 batch, Full Generate (50 steps) {mean:8.2f} +- {std:.2f} ms")

    # # Perform a batch test of 3000 samples
    # noise = T.randn((3000, 30, 3)).to("cuda")
    # mean, std = foward_time(model, noise, n=n)
    # print(f"GPU: 3000 batch, Forward pass {mean:21.2f} +- {std:.2f} ms")
    # mean, std = full_gen_time(model, noise, n=n)
    # print(f"GPU: 3000 batch, Full Generate (100 steps) {mean:8.2f} +- {std:.2f} ms")


if __name__ == "__main__":
    main()
