import os
import pickle
import pytest
from hydra import initialize, compose
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="nptyping.typing_")


from scripts.generate_data import generate_data, create_random_model
from scripts.fit_model import fit_model

from src.models.models import MeasurmentModel


@pytest.fixture
def config():
    with initialize(version_base=None, config_path="../configs"):
        cfg_generate_data = compose(config_name="generate_data", overrides=["N_timesteps=500", "local.base_dir=..", "model.r=0.1"])
        cfg_fit_model = compose(config_name="fit_model", overrides=["local.base_dir=.."])
        combined_cfg = {'cfg_generate_data': cfg_generate_data, 'cfg_fit_model': cfg_fit_model}
    return combined_cfg


def test_generate_and_print_data(config):

    config = config['cfg_generate_data']
    data_file = os.path.join(config.output_dir, 'test_data.pkl')

    timesteps = []
    x_vals = []
    z_vals = []

    with open(data_file, 'rb') as file:
        try:
            while True:
                k, x, z = pickle.load(file)
                timesteps.append(k)
                x_vals.append(x)
                z_vals.append(z)
        except EOFError:
            pass  # End of file reached

    # Plot test data
    x_values = np.array(x_vals)
    timesteps = np.array(timesteps)
    z_vals = np.array(z_vals)

    plt.figure(figsize=(10, 6))
    for i in range(x_values.shape[1]):
        plt.plot(timesteps, x_values[:, i], label=f"x[{i}]")

    plt.xlabel("Timestep")
    plt.ylabel("Value")
    plt.title("State Values Over Time")
    plt.legend()
    plt.grid(True)
    plt.show()


def test_generate_data_and_fit_model(config):
    """
    generate data with lin mes model. fit model. generate data with fitted model. compare.
    """
    cfg_generate_data = config['cfg_generate_data']
    cfg_fit_model = config['cfg_fit_model']

    ## Generate data from random matrix H
    create_random_model(cfg_generate_data)
    generate_data(cfg_generate_data)

    ## Fit model
    fit_model(cfg_fit_model)

    #load model
    fitted_model = MeasurmentModel(cfg_fit_model)

    ## Generate data from fitted model and plot

    timesteps = []
    z_gt_vals = []
    z_fitted_vals = []
    with open(cfg_generate_data.output_dir + 'test_data.pkl', 'rb') as file:
        try:
            while True:
                k, x, z_gt = pickle.load(file)
                z_fitted = fitted_model.get_measurement(x)
                timesteps.append(k)
                z_gt_vals.append(z_gt)
                z_fitted_vals.append(z_fitted)
        except EOFError:
            pass  # End of file reached

    # Plot test data
    timesteps = np.array(timesteps)
    z_gt_vals = np.array(z_gt_vals)
    z_fitted_vals = np.array(z_fitted_vals)

    plt.figure(figsize=(10, 6))

    plt.plot(timesteps, z_gt_vals[:, 0], label=f"z_gt 1 dim")
    plt.plot(timesteps, z_fitted_vals[:, 0], label=f"z_fitted 1 dim")

    plt.xlabel("Timestep")
    plt.ylabel("Value")
    plt.title("State Values Over Time")
    plt.legend()
    plt.grid(True)
    plt.show()


# if __name__ == "__main__":
#     test_generate_data_and_fit_model(config())



        