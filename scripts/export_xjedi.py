from pathlib import Path
import time

import pyrootutils

root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)

import logging

import h5py
import hydra
import numpy as np
import yaml
from omegaconf import DictConfig

from src.hydra_utils import reload_original_config

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None, config_path=str(root / "configs"), config_name="export.yaml"
)
def main(cfg: DictConfig) -> None:
    log.info("Loading run information")
    orig_cfg = reload_original_config(cfg, get_best=cfg.load)

    log.info(f"Project: {cfg.project_name}")
    log.info(f"Network Name: {cfg.network_name}")

    log.info("Loading best checkpoint")
    model_class = hydra.utils.get_class(orig_cfg.model._target_)
    model = model_class.load_from_checkpoint(orig_cfg.ckpt_path)

    log.info("Instantiating the data module for the test set")
    datamodule = hydra.utils.instantiate(orig_cfg.datamodule)
    jet_type = datamodule.hparams.data_conf.jet_type[0]

    log.info("Creating output directory.")
    outdir = Path("outputs") / jet_type
    outdir.mkdir(exist_ok=True, parents=True)

    log.info("Instantiating the trainer")
    orig_cfg.trainer["enable_progress_bar"] = True
    trainer = hydra.utils.instantiate(orig_cfg.trainer)

    # Cycle through the sampler configurations
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
            jet_labels = datamodule.test_set.high[:, -1].astype("long")
            jet_types = datamodule.jet_types()
            for i, jet_type in enumerate(jet_types):
                Path(f"outputs/{jet_type}").mkdir(exist_ok=True, parents=True)
                with h5py.File(f"outputs/{jet_type}/f"{sampler}_{steps}.h5", mode="w") as file:
                    for key in keys:
                        file.create_dataset(key, data=comb_dict[key][jet_labels == i])

    print("Done!")


if __name__ == "__main__":
    main()
