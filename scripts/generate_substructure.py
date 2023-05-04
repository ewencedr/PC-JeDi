import pyrootutils

root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)

import argparse
from argparse import Namespace
from pathlib import Path

import h5py
import numpy as np

from src.jet_substructure import dump_hlvs


def get_args() -> Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--save_dir",
        type=str,
        default="/srv/beegfs/scratch/groups/rodem/jet_diffusion/checkpoints/pc_jedi/",
        help="Path to directory where all the data is saved.",
    )
    parser.add_argument(
        "--jet_types",
        type=str,
        help="Comma separated names of the jet types to load from.",
        default="t,g,q,w,z",
    )
    parser.add_argument(
        "--file_names",
        type=str,
        default="jetnet_data.h5",
        help="Comma separated names of the files to generate from.",
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    # Paths to the relevant folders
    print(f"Evaluating Network: {args.save_dir}\n")
    path = Path(args.save_dir, "outputs")

    # Cycle through the files to generate substructure variables for each
    for jet_type in args.jet_types.split(","):
        for file_name in args.file_names.split(","):
            file_path = path / jet_type / file_name

            # Load the h5 file
            print(f"Loading {jet_type} jets from file {file_name}...")
            with h5py.File(file_path, "r") as f:
                gen_nodes = f["generated"][:]

            # Clip the eta and phi values for stability
            gen_nodes[..., 0] = np.clip(gen_nodes[..., 0], -0.5, 0.5)
            gen_nodes[..., 1] = np.clip(gen_nodes[..., 1], -0.5, 0.5)

            # Save the substructure variables in the same folder with an added suffix
            outpath = path / jet_type / (file_path.stem + "_substructure.h5")
            dump_hlvs(gen_nodes, outpath, plot=True)


if __name__ == "__main__":
    main()
