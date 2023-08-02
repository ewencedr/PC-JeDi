from pathlib import Path
import time

import pyrootutils

root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)

import logging

import h5py
import hydra
import pandas as pd
import numpy as np
import torch
import yaml
from omegaconf import DictConfig

from src.hydra_utils import reload_original_config

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path=str(root / "configs"), config_name="export.yaml")
def main(cfg: DictConfig) -> None:
    times_mean = []
    times_std = []
    solver = []
    n_particles = []
    samples = []
    batch_sizes = []
    num_runs = 1
    for num_part in [10, 30, 60, 100, 150]:  # [10, 30, 60, 100, 150]:
        torch.set_float32_matmul_precision("medium")
        log.info("Loading run information")
        orig_cfg = reload_original_config(cfg, get_best=cfg.load)

        log.info(f"Project: {cfg.project_name}")
        log.info(f"Network Name: {cfg.network_name}")

        log.info("Loading best checkpoint")
        model_class = hydra.utils.get_class(orig_cfg.model._target_)
        model = model_class.load_from_checkpoint(orig_cfg.ckpt_path)

        used_feats = (
            orig_cfg.datamodule.data_conf.jetnet_config.jet_features
            if orig_cfg.datamodule.data_conf.high_as_context
            else []
        )
        log.info("Instantiating the data module for the test set")
        if hasattr(cfg, "datamodule"):
            cfg.datamodule["ctxt_vars"] = used_feats
            cfg.datamodule.data_conf["one_hot"] = orig_cfg.datamodule.data_conf.one_hot_last
            # cfg.datamodule.data_conf[
            #    "num_particles"
            # ] = orig_cfg.datamodule.data_conf.jetnet_config.num_particles
            # log.info(
            #    f"Set number of particles to be the same as in the original config: {cfg.datamodule.data_conf.num_particles}"
            # )
            cfg.datamodule.data_conf["num_particles"] = num_part
            log.info(f"Set number of particles to {num_part}")
            datamodule = hydra.utils.instantiate(cfg.datamodule)
        else:
            datamodule = hydra.utils.instantiate(orig_cfg.datamodule)
        try:
            jet_type = datamodule.hparams.data_conf.jet_type[0]
        except AttributeError:
            jet_type = datamodule.data_conf.jet_type[0]
        log.info(datamodule)

        log.info("Creating output directory.")
        outdir = Path("outputs") / jet_type
        outdir.mkdir(exist_ok=True, parents=True)

        log.info("Instantiating the trainer")
        orig_cfg.trainer["enable_progress_bar"] = True
        trainer = hydra.utils.instantiate(orig_cfg.trainer)

        # jet_types = ["g", "q", "t", "z", "w"]
        jet_types = ["t"]
        gen_jet_types = orig_cfg.datamodule.data_conf.jetnet_config.jet_type
        label_dict = {l: i for i, l in enumerate(jet_types)}
        # Cycle through the sampler configurations
        try:
            jet_labels = datamodule.test_set.high[:, -1].astype("long")
        except AttributeError:
            jet_labels = datamodule.labels.astype("long")

        for steps in cfg.sampler_steps:
            for sampler in cfg.sampler_name:
                log.info("Setting up the generation paremeters")
                model.sampler_steps = steps
                model.sampler_name = sampler
                log.info(f"Sampler: {sampler}")
                log.info(f"Steps: {steps}")
                log.info("Running the prediction loop")
                times_temp = []
                # bs = {"10": 270000, "30": 270000, "60": 140000, "100": 80000, "150": 50000}
                bs = {"10": 270000, "30": 130000, "60": 50000, "100": 30000, "150": 16000}
                datamodule.batch_size = bs[f"{num_part}"]
                for run in range(num_runs):
                    log.info(f"RUN ---- {run} ----")
                    gen_time_track = time.time()
                    outputs = trainer.predict(model=model, datamodule=datamodule)
                    end_time = time.time() - gen_time_track
                    times_temp.append(end_time)
                log.info("Combining predictions across dataset")
                keys = list(outputs[0].keys())
                comb_dict = {key: np.vstack([o[key] for o in outputs]) for key in keys}
                log.info(f"Generation time: {end_time:.2f} s")
                print(f"comb_dict: {comb_dict[keys[0]].shape}")
                log.info("Saving HDF files.")
                times_mean.append(np.mean(times_temp))
                times_std.append(np.std(times_temp))
                solver.append(sampler)
                n_particles.append(num_part)
                samples.append(comb_dict[keys[0]].shape[0])
                batch_sizes.append(bs[f"{num_part}"])

                log.info("Saving seperate file for each jet type in test set")

                for i, jet_type in enumerate(jet_types):
                    Path(f"outputs/{jet_type}").mkdir(exist_ok=True, parents=True)
                    with h5py.File(
                        f"outputs/{jet_type}/{sampler}_{steps}_csts.h5", mode="w"
                    ) as file:
                        for key in keys:
                            file.create_dataset(
                                key, data=comb_dict[key][(jet_labels == i).squeeze()]
                            )

        print("Done!")
        print(f"times_mean: {times_mean}")
        print(f"times_std: {times_std}")
        print(f"solver: {solver}")
        print(f"n_particles: {n_particles}")
        print(f"samples: {samples}")
        print(f"batch_sizes: {batch_sizes}")
        dic = {
            "solver": solver,
            "samples": samples,
            "n_particles": n_particles,
            "times_mean": times_mean,
            "times_std": times_std,
            "batch_sizes": batch_sizes,
        }
        df = pd.DataFrame(data=dic)
        df.to_csv(Path("outputs") / "times.csv")

        print(f"Successfully saved to {Path('outputs') / 'times.csv'}")


if __name__ == "__main__":
    main()
