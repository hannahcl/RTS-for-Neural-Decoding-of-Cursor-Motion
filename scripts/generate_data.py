import hydra
from omegaconf import DictConfig, OmegaConf
import random
import numpy as np
import pickle
from pathlib import Path
import os

from src.models.gt_model import GTMeasurmentModel, GTProcessModel



@hydra.main(version_base=None, config_path="../configs", config_name="generate_data")
def generate_data(cfg : DictConfig) -> None:
    # print(OmegaConf.to_yaml(cfg.local))
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)

    random.seed(cfg.seed)
    # TODO choose models based on config , mes model lin or non lin
    model = GTProcessModel(cfg)
    mes_model = GTMeasurmentModel(cfg)

    x = np.array(cfg.gt_model.xi)
    with open(cfg.output_dir + 'train_data.pkl', 'wb') as file:
        for k in range(cfg.N_timesteps):
            x = model.step_once(x)
            # z = mes_model.get_measurement_lin(x)
            z = mes_model.get_measurement_nonlin(x)
            pickle.dump((k, x, z), file)

    x_test = np.array(cfg.gt_model.xi)
    with open(cfg.output_dir + 'test_data.pkl', 'wb') as file:
        for k in range(cfg.N_timesteps):
            x_test = model.step_once(x_test)
            # z_test = mes_model.get_measurement_lin(x_test)
            z_test = mes_model.get_measurement_nonlin(x_test)
            pickle.dump((k, x_test, z_test), file)

if __name__ == "__main__":
    generate_data()