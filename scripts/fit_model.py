import hydra
from nptyping import NDArray, Shape, Float
from omegaconf import DictConfig, OmegaConf
import random
import numpy as np
import pickle
from pathlib import Path
import os


@hydra.main(version_base=None, config_path="../configs", config_name="fit_model")
def fit_model(cfg : DictConfig) -> None:
    # print(OmegaConf.to_yaml(cfg))

    H = _fit(cfg.input_dir + '/train_data.pkl')

    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)

    np.savez(cfg.output_dir + '/fitted_model.npz', H=H)

def _fit(data_file) -> NDArray[Shape["42, 6"], Float]:

    z_values = []
    x_values = []

    with open(data_file, 'rb') as file:
        try:
            while True:
                k, x, z = pickle.load(file)
                z_values.append(z)
                x_values.append(x)
        except EOFError:
            pass  # End of file reached
    
    X = np.vstack(x_values)
    Z = np.vstack(z_values)

    # Perform least squares regression to estimate H
    H, _, _, _ = np.linalg.lstsq(X, Z, rcond=None)
    return H.T  

if __name__ == "__main__":
    fit_model()