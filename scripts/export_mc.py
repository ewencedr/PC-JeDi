"""Save the MC data in the same format as the network data for quick
comparisons and substructure calculations."""

from pathlib import Path

import h5py
from jetnet.datasets import JetNet

# How the data should be saved
save_dir = "/srv/beegfs/scratch/users/s/senguptd/jet_diffusion/"
save_name = "jetnet_data"
file_name = "jetnet_data"
type_list = ["g", "q", "t", "w", "z"]

# Cycle through each of the jet types
for jet_type in type_list:
    # Load the test dataset
    csts, high = JetNet.getData(
        jet_type=jet_type,
        data_dir="/srv/beegfs/scratch/groups/rodem/anomalous_jets/virtual_data",
        num_particles=30,
        split_fraction=[0.7, 0.0, 0.3],
        split="test",
        particle_features=["etarel", "phirel", "ptrel"],
        jet_features=["pt"],
    )

    # Conver the ptrel into pt
    csts[..., -1] *= high

    # Save in the same way as the network data is saved
    path = Path(save_dir, save_name, "outputs", jet_type)
    path.mkdir(exist_ok=True, parents=True)
    file = (path / file_name).with_suffix(".h5")
    with h5py.File(file, mode="w") as f:
        f.create_dataset("generated", data=csts)
