from pathlib import Path

import h5py
import numpy as np
import pyrootutils
from dotmap import DotMap

root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)

import torch as T

from src.plotting import plot_multi_hists_2, quantile_bins

nbins = 50
jet_types = ["t"]  # , "g", "q", "w", "z"]
plot_dir = "/srv/beegfs/scratch/users/s/senguptd/jet_diffusion/plots"

all_data = [
    {
        "label": "MC",
        "path": "/srv/beegfs/scratch/users/s/senguptd/jet_diffusion/jetnet_data",
        "file": "jetnet_data",
        "hist_kwargs": {"color": "tab:blue", "fill": True, "alpha": 0.3},
        "err_kwargs": {"color": "tab:blue", "hatch": "///"},
    },
    {
        "label": "PC-JeDi 30 EM 200",
        "path": "/srv/beegfs/scratch/users/s/senguptd/jet_diffusion/pc_jedi_30_check/2023-04-28_15-57-51-470943/",
        "file": "em_200",
        "hist_kwargs": {"color": "r"},
    },
    {
        "label": "PC-JeDi 30 DDIM 200",
        "path": "/srv/beegfs/scratch/users/s/senguptd/jet_diffusion/pc_jedi_30_check/2023-04-28_15-57-51-470943/",
        "file": "ddim_200",
        "hist_kwargs": {"color": "g"},
    },
]
all_data = [DotMap(**d) for d in all_data]

# Cycle through the jet types and variables and make each plot
for jet_type in jet_types:
    # Load the data and plot the inclusive marginalls
    for d in all_data:
        data_file = Path(d.path, "outputs", jet_type, d.file + ".h5")

        with h5py.File(data_file) as f:
            arr = f["generated"][:]
        arr[..., 0] = np.clip(arr[..., 0], -0.5, 0.5)
        arr[..., 1] = np.clip(arr[..., 1], -0.5, 0.5)
        d[jet_type] = arr
        d[jet_type + "mask"] = np.any(arr != 0, axis=-1)

    # Plot the inclusive marginals
    plot_multi_hists_2(
        data_list=[d[jet_type][d[jet_type + "mask"]] for d in all_data],
        data_labels=[d.label for d in all_data],
        col_labels=[r"$\Delta \eta$", r"$\Delta \phi$", r"$p_\mathrm{T}$"],
        hist_kwargs=[d.hist_kwargs for d in all_data],
        err_kwargs=[d.err_kwargs for d in all_data],
        bins=[
            np.linspace(-0.4, 0.4, nbins),
            np.linspace(-0.4, 0.4, nbins),
            quantile_bins(d[jet_type][..., -1].flatten(), nbins),
        ],
        do_err=True,
        legend_kwargs={
            "title": f"JetNet Data: {jet_type} jets",
            "alignment": "left",
        },
        rat_ylim=[0.5, 1.5],
        do_ratio_to_first=True,
        path=Path(plot_dir, f"{jet_type}_constituents.png"),
        do_norm=True,
    )

    # Plot the leading three constituent pt values
    pts = [d[jet_type][..., -1] for d in all_data]
    top3 = [T.topk(T.tensor(pt), 3, dim=-1).values for pt in pts]
    plot_multi_hists_2(
        data_list=top3,
        data_labels=[d.label for d in all_data],
        col_labels=[
            r"Leading constituent $p_\mathrm{T}$",
            r"2nd leading constituent $p_\mathrm{T}$",
            r"3rd leading constituent $p_\mathrm{T}$",
        ],
        hist_kwargs=[d.hist_kwargs for d in all_data],
        err_kwargs=[d.err_kwargs for d in all_data],
        bins=quantile_bins(top3[0], nbins, axis=(0)).T.tolist(),
        do_err=True,
        legend_kwargs={
            "title": f"JetNet Data: {jet_type} jets",
            "alignment": "left",
        },
        rat_ylim=[0.5, 1.5],
        do_ratio_to_first=True,
        path=Path(plot_dir, f"{jet_type}_leading_constituents.png"),
        do_norm=True,
    )
