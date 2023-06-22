import pyrootutils
import yaml

root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)

from utils import standard_job_array


def main() -> None:
    """Main executable script."""

    standard_job_array(
        job_name="epic_30",
        work_dir="/home/users/s/senguptd/UniGe/generation/diffusion/PC-JeDi",
        image_path="/home/users/s/senguptd/UniGe/generation/diffusion/PC-JeDi/container/diffusion.sif",
        command="python scripts/train.py",
        log_dir="/home/users/s/senguptd/UniGe/generation/diffusion/PC-JeDi/jobs",
        n_gpus=1,
        n_cpus=4,
        time_hrs=24,
        mem_gb=25,
        script_save_path="/home/users/s/senguptd/UniGe/generation/diffusion/PC-JeDi/",
        double_dash=False,
        rename_dict={"project_name": "project", "experiment": "experiment", "datamodule.loader_kwargs.batch_size": "batch_size", "datamodule.data_conf.recalculate_jet_from_pc": "recalculate_jet_from_pc", "datamodule.data_conf.log_squash_pt": "log_squash_pt", "model.epic_jedi_config.latent": "latent"},
        opt_dict={
            "project_name": "epic_jedi1_c",
            "experiment": "epic_30_cond.yaml",
            "datamodule.loader_kwargs.batch_size": [1024, 256],
            "datamodule.data_conf.recalculate_jet_from_pc": [True, False],
            "datamodule.data_conf.log_squash_pt": [True, False],
            "model.epic_jedi_config.latent": [5, 16]
        },
    )


if __name__ == "__main__":
    main()
