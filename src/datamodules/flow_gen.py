from pathlib import Path
import h5py
import numpy as np
import torch as T
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset

from src.utils import onehot_encode

DATA_BOUNDS = {
    "pt": [159.33, 3156.72],
    "eta": [-2.7, 2.7],
    "mass": [3.35, 573.61],
    "num_particles": [5, 150],
    "type": [0, 4],
}


class FlowGeneratedData(LightningDataModule):
    def __init__(self, flow_path, ctxt_vars, batch_size, data_conf):
        super().__init__()
        self.batch_size = batch_size
        self.data_conf = data_conf

        # Load the data generated using the flow
        with h5py.File(Path(flow_path)) as f:
            list_of_keys = list(f.keys())[1:]
            all_high = np.hstack([f[c][:] for c in list_of_keys])

        print(f"Cropping data: {len(all_high)}")
        # Make sure that the data does not lie outside the bounds
        mask = np.full(len(all_high), True)
        for i, c in enumerate(list_of_keys):
            v_min = DATA_BOUNDS[c][0]
            v_max = DATA_BOUNDS[c][1]
            v_mask = (v_min <= all_high[:, i]) & (all_high[:, i] <= v_max)
            mask = mask & v_mask
            print(f" - invalid {c} = {np.sum(~v_mask)}")
        all_high = all_high[mask]
        print(f"Total remaining: {len(all_high)}")

        # Our model expects pt to be seperate
        all_pt = all_high[..., list_of_keys.index("pt"), None]

        # Create the mask
        num_particles = all_high[..., list_of_keys.index("num_particles"), None]
        # clamp the num_particles to the that of data_conf
        num_particles = np.clip(num_particles, 5, data_conf["num_particles"])

        max_n = int(max(num_particles))
        all_mask = np.arange(0, max_n)[None, :]
        all_mask = np.broadcast_to(all_mask, (len(all_high), max_n))
        all_mask = all_mask < num_particles

        # Only keep the features as requested in ctxt vars
        if ctxt_vars:
            ctxt_high = np.hstack(
                [all_high[..., list_of_keys.index(c)].reshape(-1, 1) for c in ctxt_vars]
            )
        else:
            print("No context variables specified, using None")
            jet_type = all_high[:, -1].astype("int")
            ctxt_high = jet_type
        # Onehot encode the jet type
        jet_type = all_high[:, -1].astype("int")
        jet_onehot = onehot_encode(jet_type, max_idx=4)  # 4 to match jetnet types
        jet_type = jet_type[:, None].astype(all_high.dtype)
        if data_conf["one_hot"]:
            ctxt_high = np.hstack([ctxt_high[..., :-1], jet_onehot, jet_type])

        # Need to be accessed later
        self.mask = all_mask
        self.high = ctxt_high
        self.pt = all_pt
        self.labels = jet_type

        # Create a dataloader for this to work (called the test set for the model)
        self.test_set = TensorDataset(
            T.from_numpy(self.pt),  # Using this as a placeholder it is ignored
            T.from_numpy(self.mask),
            T.from_numpy(self.high),
            T.from_numpy(self.pt),
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_set, batch_size=self.batch_size, pin_memory=True, num_workers=128
        )
