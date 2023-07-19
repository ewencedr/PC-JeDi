from pathlib import Path
import time

import pyrootutils

root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)

import logging

import h5py
import hydra
import numpy as np
import torch
import yaml
from omegaconf import DictConfig

from src.hydra_utils import reload_original_config

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path=str(root / "configs"), config_name="export.yaml")
def main(cfg: DictConfig) -> None:
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
        cfg.datamodule.data_conf[
            "num_particles"
        ] = orig_cfg.datamodule.data_conf.jetnet_config.num_particles
        log.info(
            f"Set number of particles to be the same as in the original config: {cfg.datamodule.data_conf.num_particles}"
        )
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

    jet_types = ["g", "q", "t", "z", "w"]
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

            log.info("Running the prediction loop")
            gen_time_track = time.time()
            outputs = trainer.predict(model=model, datamodule=datamodule)
            end_time = time.time() - gen_time_track
            log.info("Combining predictions across dataset")
            keys = list(outputs[0].keys())
            comb_dict = {key: np.vstack([o[key] for o in outputs]) for key in keys}
            log.info(f"Generation time: {end_time:.2f} s")

            log.info("Saving HDF files.")

            log.info("Saving seperate file for each jet type in test set")

            for i, jet_type in enumerate(jet_types):
                Path(f"outputs/{jet_type}").mkdir(exist_ok=True, parents=True)
                with h5py.File(f"outputs/{jet_type}/{sampler}_{steps}_csts.h5", mode="w") as file:
                    for key in keys:
                        file.create_dataset(key, data=comb_dict[key][(jet_labels == i).squeeze()])

    print("Done!")


if __name__ == "__main__":
    main()
