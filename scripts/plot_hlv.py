import pyrootutils

root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)
from pathlib import Path
import h5py
import numpy as np
from dotmap import DotMap


from src.plotting import plot_multi_correlations, plot_multi_hists_2


n_kde_points = 200
jet_types = ["t"]  # , "g", "q", "w", "z"]
sub_vars = [
    # "tau1",
    # "tau2",
    # "tau3",
    "tau21",
    "tau32",
    # "d12",
    # "d23",
    # "ecf2",
    # "ecf3",
    "d2",
    "mass",
    "pt",
]
feat_spread_vars = ["tau21", "tau32", "d2", "mass"]

num_const = 30
project = "epic_jedi1_c"
directory = "2023-06-17_13-02-33-405488"
nbins = 50
jet_types = ["t"]  # , "g", "q", "w", "z"]
plot_dir = (
    f"/srv/beegfs/scratch/users/s/senguptd/jet_diffusion/plots/{num_const}_constituents"
)

if not Path(plot_dir).exists():
    Path(plot_dir).mkdir(parents=True)


all_data = [
    {
        "label": "MC",
        "path": f"/srv/beegfs/scratch/users/s/senguptd/jet_diffusion/jetnet_data/jetnet_data_{num_const}",
        "file": f"jetnet_data_test",
        "hist_kwargs": {"color": "tab:blue", "fill": True, "alpha": 0.3},
        "err_kwargs": {"color": "tab:blue", "hatch": "///"},
    },
    {
        "label": f"EPiC-JeDi {num_const} EM 200",
        "path": f"/srv/beegfs/scratch/users/s/senguptd/jet_diffusion/{project}/{directory}",
        "file": "em_200_csts",
        "hist_kwargs": {"color": "r"},
    },
    {
        "label": f"EPiC-JeDi {num_const} DDIM 200",
        "path": f"/srv/beegfs/scratch/users/s/senguptd/jet_diffusion/{project}/{directory}",
        "file": "ddim_200_csts",
        "hist_kwargs": {"color": "g"},
    },
    # {
    #     "label": f"EPiC-JeDi {num_const} Midpoint 50",
    #     "path": f"/srv/beegfs/scratch/users/s/senguptd/jet_diffusion/{project}/{directory}",
    #     "file": "midpoint_50_csts",
    #     "hist_kwargs": {"color": "b"},
    # },
    {
        "label": f"EPiC-JeDi {num_const} Midpoint 200",
        "path": f"/srv/beegfs/scratch/users/s/senguptd/jet_diffusion/{project}/{directory}",
        "file": "midpoint_200_csts",
        "hist_kwargs": {"color": "b", "ls": "--"},
    },
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
    for sub_var in sub_vars:
        sub_data = []
        for d in all_data:
            data_file = Path(d.path, "outputs", jet_type, d.file + "_substructure.h5")
            with h5py.File(data_file) as f:
                sub_data.append(f[sub_var][:][:, None])

        plot_multi_hists_2(
            data_list=sub_data,
            data_labels=[d.label for d in all_data],
            col_labels=[sub_var],
            hist_kwargs=[d.hist_kwargs for d in all_data],
            err_kwargs=[d.err_kwargs for d in all_data],
            bins=np.linspace(*np.quantile(sub_data[0], [0.001, 0.999]), nbins),
            do_err=True,
            legend_kwargs={
                "title": f"JetNet Data: {jet_type} jets",
                "alignment": "left",
            },
            do_ratio_to_first=True,
            path=Path(plot_dir, f"{jet_type}_{sub_var}.pdf"),
            do_norm=True,
        )

    # For top jets it is also interesting to plot the seperate mass windows
    if jet_type == "t":
        masses = []
        for d in all_data:
            data_file = Path(d.path, "outputs", jet_type, d.file + "_substructure.h5")
            with h5py.File(data_file) as f:
                masses.append(f["mass"][:])

        # Select based on the w mass window and plot the substructure
        w_idxes = [(mass > 60) & (mass < 100) for mass in masses]
        for sub_var in sub_vars:
            sub_data = []
            for i, d in enumerate(all_data):
                data_file = Path(
                    d.path, "outputs", jet_type, d.file + "_substructure.h5"
                )
                with h5py.File(data_file) as f:
                    sub_data.append(f[sub_var][w_idxes[i]][:, None])

            plot_multi_hists_2(
                data_list=sub_data,
                data_labels=[d.label for d in all_data],
                col_labels=[sub_var],
                hist_kwargs=[d.hist_kwargs for d in all_data],
                err_kwargs=[d.err_kwargs for d in all_data],
                bins=np.linspace(*np.quantile(sub_data[0], [0.001, 0.999]), nbins),
                do_err=True,
                legend_kwargs={
                    "title": rf"JetNet Data: {jet_type} jets where $m_j \in [60, 100]$ GeV",
                    "alignment": "left",
                },
                do_ratio_to_first=True,
                path=Path(plot_dir, f"{jet_type}_{sub_var}_60_100.pdf"),
                do_norm=True,
            )

        # Select based on the w mass window and plot the substructure
        t_idxes = [(mass > 140) & (mass < 200) for mass in masses]
        for sub_var in sub_vars:
            sub_data = []
            for i, d in enumerate(all_data):
                data_file = Path(
                    d.path, "outputs", jet_type, d.file + "_substructure.h5"
                )
                with h5py.File(data_file) as f:
                    sub_data.append(f[sub_var][t_idxes[i]][:, None])

            plot_multi_hists_2(
                data_list=sub_data,
                data_labels=[d.label for d in all_data],
                col_labels=[sub_var],
                hist_kwargs=[d.hist_kwargs for d in all_data],
                err_kwargs=[d.err_kwargs for d in all_data],
                bins=np.linspace(*np.quantile(sub_data[0], [0.001, 0.999]), nbins),
                do_err=True,
                legend_kwargs={
                    "title": rf"JetNet Data: {jet_type} jets where $m_j \in [60, 100]$ GeV",
                    "alignment": "left",
                },
                do_ratio_to_first=True,
                path=Path(plot_dir, f"{jet_type}_{sub_var}_140_200.pdf"),
                do_norm=True,
            )

    # Now for plotting the feature spreads
    # Cycle through the data list
    for d in all_data:
        # Load the requested substructure variables
        data_file = Path(d.path, "outputs", jet_type, d.file + "_substructure.h5")
        with h5py.File(data_file) as f:
            for s in feat_spread_vars:
                d[s] = f[s][:]

    # Combine the columns to pass to the plotter
    # plot_multi_correlations(
    #     data_list=[np.stack([d[s] for s in feat_spread_vars]).T for d in all_data],
    #     data_labels=[d.label for d in all_data],
    #     col_labels=feat_spread_vars,
    #     n_bins=nbins,
    #     n_kde_points=n_kde_points,
    #     hist_kwargs=[d.hist_kwargs for d in all_data],
    #     err_kwargs=[d.err_kwargs for d in all_data],
    #     legend_kwargs={
    #         "loc": "upper right",
    #         "alignment": "right",
    #         "fontsize": 15,
    #         "bbox_to_anchor": (0.8, 0.90),
    #     },
    #     path=Path(plot_dir, f"hlv_corr_{jet_type}.pdf"),
    # )
