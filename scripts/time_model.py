import hydra
from omegaconf import OmegaConf
import pyrootutils
import yaml

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


def full_gen_time(model, noise, n=10, mask=None, context=None) -> float:
    t = []
    mask = T.full(noise.shape[:-1], True, device=noise.device)
    for _ in range(n + 1):
        ts = time.process_time()
        euler_maruyama_sampler(
            model=model,
            diff_sched=model.diff_sched,
            mask=mask,
            initial_noise=noise,
            ctxt=context,
            n_steps=200,
        )
        te = time.process_time()
        t.append(te - ts)
    return np.mean(t[1:]) * 1000, np.std(t[1:]) * 1000


def foward_time(model, noise, labels=None, n=10, context=None, mode="jetdiff") -> float:
    t = []
    mask = T.full(noise.shape[:-1], True, device=noise.device)
    diff_time = T.rand(len(noise), device=noise.device)
    for _ in range(n + 1):
        ts = time.process_time()
        if mode == "jetdiff":
            model.forward(noise, diff_time, mask, context)
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
        default="/srv/beegfs/scratch/users/s/senguptd/jet_diffusion/epic_jedi1_c",
        help="Path to directory where the model and its checkpoints are saved.",
    )
    parser.add_argument(
        "--jetdiff-model-name",
        type=str,
        default="2023-06-06_15-48-57-904902",
        help="Name of the model to load for generation.",
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
    n = 25
    # Load the full model configuration
    full_path = Path(args.jetdiff_model_dir, args.jetdiff_model_name)
    original_config = OmegaConf.load(full_path / "full_config.yaml")
    if original_config.datamodule.data_conf.high_as_context:
        context_dim = len(original_config.datamodule.data_conf.jet_features)
    else:
        context_dim = 0
    # Load the checkpoint
    ckpt_path = find_best_checkpoint(full_path)
    num_const = original_config.datamodule.data_conf.num_particles
    model_class = hydra.utils.get_class(original_config.model._target_)
    model = model_class.load_from_checkpoint(ckpt_path)

    print(".:JeDi (EM-200):.")

    # Create the dummy input and calculate
    noise = T.randn((1, num_const, 3)).to("cpu")
    context = T.randn((1, context_dim)).to("cpu") if context_dim > 0 else None
    mean_cpu_forward, std_cpu_forward = foward_time(
        model.to("cpu"), noise, context=context, n=n
    )
    print(f"CPU: Forward pass {mean_cpu_forward:33.2f} +- {std_cpu_forward:.2f} ms")
    mean_cpu_fullgen, std_cpu_fullgen = full_gen_time(
        model.to("cpu"), noise, context=context, n=n
    )
    print(
        f"CPU: Full Generate (200 steps) {mean_cpu_fullgen:20.2f} +- {std_cpu_fullgen:.2f} ms"
    )

    # Move the the test to the gpu
    model = model.to("cuda")
    noise = noise.to("cuda")
    context = context.to("cuda")
    mean_gpu_forward, std_gpu_forward = foward_time(model, noise, context=context, n=n)
    print(f"GPU: Forward pass {mean_gpu_forward:33.2f} +- {std_gpu_forward:.2f} ms")
    mean_gpu_fullgen, std_gpu_fullgen = full_gen_time(
        model, noise, context=context, n=n
    )
    print(
        f"GPU: Full Generate (200 steps) {mean_gpu_fullgen:20.2f} +- {std_gpu_fullgen:.2f} ms"
    )

    # Perform a batch test of 10 samples
    noise = T.randn((10, num_const, 3)).to("cuda")
    context = T.randn((10, context_dim)).to("cuda") if context_dim > 0 else None
    mean_gpu_forward10, std_gpu_forward10 = foward_time(
        model, noise, context=context, n=n
    )
    print(
        f"GPU: 10 batch, Forward pass {mean_gpu_forward10:23.2f} +- {std_gpu_forward10:.2f} ms"
    )
    mean_gpu_fullgen10, std_gpu_fullgen10 = full_gen_time(
        model, noise, context=context, n=n
    )
    print(
        f"GPU: 10 batch, Full Generate (200 steps) {mean_gpu_fullgen10:10.2f} +- {std_gpu_fullgen10:.2f} ms"
    )

    # Perform a batch test of 1000 samples
    noise = T.randn((1000, num_const, 3)).to("cuda")
    context = T.randn((1000, context_dim)).to("cuda") if context_dim > 0 else None
    mean_gpu_forward1000, std_gpu_forward1000 = foward_time(
        model, noise, context=context, n=n
    )
    print(
        f"GPU: 1000 batch, Forward pass {mean_gpu_forward1000:21.2f} +- {std_gpu_forward1000:.2f} ms"
    )
    mean_gpu_fullgen1000, std_fullgen1000 = full_gen_time(
        model, noise, context=context, n=n
    )
    print(
        f"GPU: 1000 batch, Full Generate (200 steps) {mean_gpu_fullgen1000:8.2f} +- {std_fullgen1000:.2f} ms"
    )

    print("TORCH VERSION:", T.__version__)
    print("CUDA VERSION:", T.version.cuda)
    print("CUDNN VERSION:", T.backends.cudnn.version())
    print("GPU MODEL:", T.cuda.get_device_name(0))

    time_dict = {
        "TORCH": T.__version__,
        "CUDA": T.version.cuda,
        "CUDNN": T.backends.cudnn.version(),
        "GPU_MODEL": T.cuda.get_device_name(0),
        "cpu_forward": mean_cpu_forward.tolist(),
        "cpu_forward_std": std_cpu_forward.tolist(),
        "cpu_fullgen": mean_cpu_fullgen.tolist(),
        "cpu_fullgen_std": std_cpu_fullgen.tolist(),
        "gpu_forward": mean_gpu_forward.tolist(),
        "gpu_forward_std": std_gpu_forward.tolist(),
        "gpu_fullgen": mean_gpu_fullgen.tolist(),
        "gpu_fullgen_std": std_gpu_fullgen.tolist(),
        "gpu_forward10": mean_gpu_forward10.tolist(),
        "gpu_forward10_std": std_gpu_fullgen10.tolist(),
        "gpu_fullgen10": mean_gpu_fullgen10.tolist(),
        "gpu_fullgen10_std": std_gpu_fullgen10.tolist(),
        "gpu_forward1000": mean_gpu_forward1000.tolist(),
        "gpu_forward1000_std": std_gpu_forward1000.tolist(),
        "gpu_fullgen1000": mean_gpu_fullgen1000.tolist(),
        "gpu_fullgen1000_std": std_fullgen1000.tolist(),
    }
    with open(full_path / "time_dict.yaml", "w") as f:
        yaml.dump(time_dict, f)


if __name__ == "__main__":
    main()
