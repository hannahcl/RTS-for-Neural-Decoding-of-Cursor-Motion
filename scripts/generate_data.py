import hydra
from omegaconf import DictConfig, OmegaConf
import random
import numpy as np
import pickle
from pathlib import Path
import os

from src.models.models import MeasurmentModel, ProcessModel


@hydra.main(version_base=None, config_path="../configs", config_name="generate_data")
def generate_data(cfg : DictConfig) -> None:

    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)

    random.seed(cfg.seed)
    model = ProcessModel(cfg)
    mes_model = MeasurmentModel(cfg)

    x = np.array(cfg.xi)
    with open(cfg.output_dir + 'train_data.pkl', 'wb') as file:
        for k in range(cfg.N_timesteps_train):
            x = model.step_once(x)
            z = mes_model.get_measurement(x)
            pickle.dump((k, x, z), file)

            if k%10 == 0:
                x[4] = 0
                x[5] = 10

            if k%30 == 0:
                x[4] = -10
                x[5] = 0

            if k%60 == 0:
                x[4] = 10
                x[5] = 0

            if k%80 == 0:
                x[4] = 0
                x[5] = -10

    x_test = np.array(cfg.xi)
    with open(cfg.output_dir + 'test_data.pkl', 'wb') as file:
        for k in range(cfg.N_timesteps_test):
            x_test = model.step_once(x_test)
            z_test = mes_model.get_measurement(x_test)
            pickle.dump((k, x_test, z_test), file)

            if k%20 == 0:
                x_test[4] = 10
                x_test[5] = 0

            if k%40 == 0:
                x_test[4] = 0
                x_test[5] = 10

            if k%60 == 0:
                x_test[4] = -10
                x_test[5] = 0

            if k%80 == 0:
                x_test[4] = 0
                x_test[5] = -10

def create_random_model(cfg: DictConfig) -> None:
    H1 = np.random.rand(cfg.model.C, cfg.model.nx)
    H2 = np.random.rand(cfg.model.C, cfg.model.nx)

    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)

    np.savez(cfg.path_to_measurement_model, H1=H1, H2=H2)


if __name__ == "__main__":
    create_random_model()
    generate_data()