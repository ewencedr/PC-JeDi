import pyrootutils

root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)

import argparse
from argparse import Namespace
from pathlib import Path

import h5py
import numpy as np
import yaml
from jetnet.datasets import JetNet
from jetnet.evaluation import cov_mmd, fpnd, w1efp, w1m, w1p

from src.numpy_utils import undo_log_squash
from src.physics import numpy_locals_to_mass_and_pt


def get_args() -> Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        help="Name of the model to evaluate.",
        default="/srv/beegfs/scratch/users/s/senguptd/jet_diffusion/epic_jedi_30/2023-05-19_10-03-09-602489",
    )
    parser.add_argument(
        "--num_eval_samples",
        type=int,
        help="Number of samples per batch in the bootsrapped evaluation methods.",
        default=10000,
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        help="Number of bootstrapped batches to evaluate.",
        default=40,
    )
    args = parser.parse_args()
    return args


def main() -> None:
    args = get_args()

    # Use the info yaml file to tell which jetnet data to pull from
    with open(args.model_name + "/full_config.yaml") as file:
        info = yaml.safe_load(file)
    # Paths to the relevant folders

    jet_type = info["datamodule"]["data_conf"]["jet_type"]
    num_particles = info["datamodule"]["data_conf"]["num_particles"]
    print(f"Evaluating Network: {args.model_name}\n")
    paths = [Path(args.model_name, "outputs", jet) for jet in jet_type]
    for path in paths:
        if not path.exists():
            # create the path
            path.mkdir(parents=True, exist_ok=True)

    # Load the test jetnet data
    for path, jets in zip(paths, jet_type):
        real_nodes, real_jets = JetNet.getData(
            jet_type=jets,
            data_dir="/srv/beegfs/scratch/groups/rodem/anomalous_jets/virtual_data",
            split="test",
            split_fraction=[0.7, 0.0, 0.3],
            num_particles=num_particles,
            particle_features=["etarel", "phirel", "ptrel"],
            jet_features=["pt", "mass"],
        )
        real_mask = np.any(real_nodes != 0, axis=-1)

        # Get the (pt_relative) point cloud mass and pt
        real_pc_jet = numpy_locals_to_mass_and_pt(real_nodes, real_mask)

        # Load the generated jets from the saved h5 files.
        # Glob only files that are of the form sampler_steps.h5 and NOT sampler_steps_substrucutre.h5
        h5_files = list(path.glob("*.h5"))
        h5_files = [file for file in h5_files if len(file.stem.split("_")) == 2]
        h5_files.sort()

        # Make the directory which will hold all output yaml files
        yaml_score_dir = path / "yaml_scores"
        yaml_score_dir.mkdir(parents=True, exist_ok=True)

        # Cycle through the HDF files and evaluate the jets within
        for file in h5_files:
            # Get the name of the sampler from the file name
            sampler = file.stem

            print("=" * 60)
            print("=" * 60)
            print(f"Evaluating {sampler} sampler\n")

            # Load the data from the h5 file
            with h5py.File(file, "r") as f:
                gen_nodes = f["etaphipt_frac"][:]

            # Fix the data by clipping
            gen_nodes[..., 0] = np.clip(gen_nodes[..., 0], -0.5, 0.5)
            gen_nodes[..., 1] = np.clip(gen_nodes[..., 1], -0.5, 0.5)
            gen_nodes[..., 2] = np.clip(gen_nodes[..., 2], 0, 1.0)

            # Generate a mask for the nodes (should be identical to real_mask)
            gen_mask = ~np.all(gen_nodes == 0, axis=-1)

            # Assert a perfect 1 to 1 matching with the real data
            assert np.all(gen_mask == real_mask)

            # Get the generated point cloud pt and invariant masses
            gen_pc_jet = numpy_locals_to_mass_and_pt(gen_nodes, gen_mask)

            # Define the keyword arguments for the bootstrapping
            bootstrap = {
                "num_eval_samples": args.num_eval_samples,
                "num_batches": args.num_batches,
            }

            # Calculate all the scores on the generated jets
            if num_particles == 30:
                fpnd_val = fpnd(
                    gen_nodes,
                    jet_type=jets,
                )
            else:
                fpnd_val = -999
            print(f"FPND:  {fpnd_val:4.3E}")

            w1m_val, w1m_err = w1m(real_nodes, gen_nodes, **bootstrap)
            print(f"W1M:   {w1m_val:4.3E} +- {w1m_err:5.4E}")

            w1p_val, w1p_err = w1p(real_nodes, gen_nodes, **bootstrap)
            w1p_val = w1p_val.mean()
            w1p_err = w1p_err.mean()
            print(f"W1P:   {w1p_val:4.3E} +- {w1p_err:5.4E}")

            w1efp_val, w1efp_err = w1efp(real_nodes, gen_nodes, **bootstrap, efp_jobs=1)
            w1efp_val = w1efp_val.mean()
            w1efp_err = w1efp_err.mean()
            print(f"W1EFP: {w1efp_val:4.3E} +- {w1efp_err:5.4E}")

            cov, mmd = cov_mmd(
                real_nodes,
                gen_nodes,
                num_eval_samples=200,
                num_batches=args.num_batches,
            )
            print(f"COV:   {(cov):4.3E}")
            print(f"MMD:   {mmd:4.3E}")

            pt_corr = np.mean(np.abs(real_pc_jet[..., 0] - gen_pc_jet[..., 0]))
            mass_corr = np.mean(np.abs(real_pc_jet[..., 1] - gen_pc_jet[..., 1]))
            print(f"pt_mae   {pt_corr:4.3E}")
            print(f"m_mae:   {mass_corr:4.3E}")

            # Save the scores in a dictionary and then to a yaml file
            with open(yaml_score_dir / sampler, "w") as f:
                yaml.dump(
                    {
                        "fpnd": float(fpnd_val),
                        "w1m": float(w1m_val),
                        "w1m_err": float(w1m_err),
                        "w1p": float(w1p_val),
                        "w1p_err": float(w1p_err),
                        "w1efp": float(w1efp_val),
                        "w1efp_err": float(w1efp_err),
                        "cov": float(cov),
                        "mmd": float(mmd),
                        "pt_mae": float(pt_corr),
                        "m_mae": float(mass_corr),
                    },
                    f,
                )


if __name__ == "__main__":
    main()
