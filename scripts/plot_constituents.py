import pyrootutils

root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)
from pathlib import Path

import h5py
import numpy as np
import torch as T
from dotmap import DotMap

from src.plotting import plot_multi_hists_2, quantile_bins


#========== CONFIGURATION ==========
num_const = 30
epic_jedi_project = "final_cedric_changes"
epicjedi_directory = "2023-07-16_09-58-26-914027"


nbins = 50
jet_types = ["t"]  # , "g", "q", "w", "z"]
plot_dir = (
    f"/srv/beegfs/scratch/users/s/senguptd/jet_diffusion/plots/{num_const}_constituents"
)

eta_range = (-1.5, 1.0) if num_const == 150 else (-0.5, 0.5)
phi_range = (-0.5, 0.5)

if not Path(plot_dir).exists():
    Path(plot_dir).mkdir(parents=True)

#========== DATA and directory config ==========
all_data = [
    {
        "label": "MC",
        "path": f"/srv/beegfs/scratch/users/s/senguptd/jet_diffusion/jetnet_data/jetnet_data_{num_const}",
        "file": f"jetnet_data_test_csts",
        "hist_kwargs": {"color": "tab:blue", "fill": True, "alpha": 0.3},
        "err_kwargs": {"color": "tab:blue", "hatch": "///"},
    },
    {
        "label": f"EPiC-JeDi {num_const} EM 200",
        "path": f"/srv/beegfs/scratch/users/s/senguptd/jet_diffusion/{epic_jedi_project}/{epicjedi_directory}",
        "file": "em_200_csts",
        "hist_kwargs": {"color": "r"},
    },
    # {
    #     "label": f"EPiC-JeDi {num_const} DDIM 200",
    #     "path": f"/srv/beegfs/scratch/users/s/senguptd/jet_diffusion/{epic_jedi_project}/{epicjedi_directory}",
    #     "file": "ddim_200_csts",
    #     "hist_kwargs": {"color": "g"},
    # },
    # # {
    # #     "label": f"EPiC-JeDi {num_const} Midpoint 50",
    # #     "path": f"/srv/beegfs/scratch/users/s/senguptd/jet_diffusion/{project}/{directory}",
    # #     "file": "midpoint_50_csts",
    # #     "hist_kwargs": {"color": "b"},
    # # },
    # {
    #     "label": f"EPiC-JeDi {num_const} Midpoint 200",
    #     "path": f"/srv/beegfs/scratch/users/s/senguptd/jet_diffusion/{epic_jedi_project}/{epicjedi_directory}",
    #     "file": "midpoint_100_csts",
    #     "hist_kwargs": {"color": "b", "ls": "--"},
    # },
    # {
    #     "label": "PC-JeDi 150 EM 200",
    #     "path": "/srv/beegfs/scratch/users/s/senguptd/jet_diffusion/pc_jedi_150/2023-05-19_11-01-28-016315",
    #     "file": "em_200",
    #     "hist_kwargs": {"color": "r", "ls": "--"},
    # },
    # {
    #     "label": "PC-JeDi 150 DDIM 200",
    #     "path": "/srv/beegfs/scratch/users/s/senguptd/jet_diffusion/pc_jedi_150/2023-05-19_11-01-28-016315",
    #     "file": "ddim_200",
    #     "hist_kwargs": {"color": "g", "ls": "--"},
    # },
]
all_data = [DotMap(**d) for d in all_data]

# Cycle through the jet types and variables and make each plot
for jet_type in jet_types:
    # Load the data and plot the inclusive marginalls
    for d in all_data:
        data_file = Path(d.path, "outputs", jet_type, d.file + ".h5")

        with h5py.File(data_file) as f:
            arr = f["etaphipt_frac"][:]
        arr[..., 0] = np.clip(arr[..., 0], eta_range[0], eta_range[1])
        arr[..., 1] = np.clip(arr[..., 1], phi_range[0], phi_range[1])
        d[jet_type] = arr
        d[jet_type + "mask"] = np.any(arr != 0, axis=-1)


#========== PLOTTING ==========
    # Plot the inclusive marginals
    plot_multi_hists_2(
        data_list=[d[jet_type][d[jet_type + "mask"]] for d in all_data],
        data_labels=[d.label for d in all_data],
        col_labels=[r"$\Delta \eta$", r"$\Delta \phi$", r"$\frac{p_\mathrm{T}}{Jet p_\mathrm{T}}$"],
        hist_kwargs=[d.hist_kwargs for d in all_data],
        err_kwargs=[d.err_kwargs for d in all_data],
        bins=[
            np.linspace(eta_range[0], eta_range[1], nbins),
            np.linspace(phi_range[0], phi_range[1], nbins),
            quantile_bins(d[jet_type][..., -1].flatten(), nbins),
        ],
        do_err=True,
        legend_kwargs={
            "title": f"JetNet Data: {jet_type} jets",
            "alignment": "left",
            "frameon": False,
        },
        rat_ylim=[0.5, 1.5],
        do_ratio_to_first=True,
        path=Path(plot_dir, f"{jet_type}_constituents.pdf"),
        do_norm=True,
        logy=True,
        ypad=.05,

    )

    # Plot the leading three constituent pt values
    pts = [d[jet_type][..., -1] for d in all_data]
    top20 = [
            T.sort(T.tensor(pt), dim=-1, descending=True)[0][:, [0, 4, 19]]
            for pt in pts
        ]
    plot_multi_hists_2(
        data_list=top20,
        data_labels=[d.label for d in all_data],
        col_labels=[
            r"Leading constituent $\frac{p_\mathrm{T}}{Jet p_\mathrm{T}}$",
            r"$5^{th}$ leading constituent $\frac{p_\mathrm{T}}{Jet p_\mathrm{T}}$",
            r"$20^{th}$ leading constituent $\frac{p_\mathrm{T}}{Jet p_\mathrm{T}}$",
        ],
        hist_kwargs=[d.hist_kwargs for d in all_data],
        err_kwargs=[d.err_kwargs for d in all_data],
        bins=quantile_bins(top20[0], nbins, axis=(0)).T.tolist(),
        do_err=True,
        legend_kwargs={
            "title": f"JetNet Data: {jet_type} jets",
            "alignment": "left",
            "frameon": False,
        },
        rat_ylim=[0.5, 1.5],
        do_ratio_to_first=True,
        path=Path(plot_dir, f"{jet_type}_leading_constituents.pdf"),
        do_norm=True,
        logy=True,
        ypad=.35,
    )
