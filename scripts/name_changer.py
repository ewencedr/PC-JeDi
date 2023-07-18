from pathlib import Path
import glob

network = "2023-07-10_18-56-53-297525"
path = Path(f"/srv/beegfs/scratch/users/s/senguptd/jet_diffusion/final_cedric_changes/{network}/outputs/t")

files = list(glob.glob(str(path / "*.h5")))
names = [Path(file).name for file in files]
for item in files:
    item = Path(item)
    if "substructure" in item.name:
        print(f"Skipping")
    else:
        name = item.name.split(".")[0]
        name = name+"_csts.h5"
        item.rename(path / name)
        print(item.name.split("."))
print(files)