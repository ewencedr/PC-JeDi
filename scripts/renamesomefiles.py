from pathlib import Path
import h5py
import numpy as np


key_dict = {"etaphipt": "etaphipt_frac", "etaphiptfrac":"etaphipt"}
model = "150"
PATH = Path(f"/home/users/s/senguptd/scratch/jet_diffusion/epic_fm/{model}/outputs/t/")

h5sinpath = list(PATH.glob("*csts.h5"))

for h5path in h5sinpath:
    # Each file either has "etaphipt" and "etaphiptfrac" or just "etaphipt"
    # we want to rename the key "etaphipt" to "etaphipt_frac" and "etaphiptfrac" to "etaphipt"
    # moving causes corruption, so we need to copy and then delete
    with h5py.File(h5path, "r+") as f:
        if "etaphipt" in f.keys():
            f.copy("etaphipt", "etaphipt_frac")
            del f["etaphipt"]
        if "etaphiptfrac" in f.keys():
            f.copy("etaphiptfrac", "etaphipt")
            del f["etaphiptfrac"]
        print(f"Renamed keys in {h5path}")