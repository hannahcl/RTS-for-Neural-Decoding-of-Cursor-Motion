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
from scripts.run_filter import run_filters

from src.models.models import MeasurmentModel
from src.analysis.plots import \
    plot_measurement_and_prediction, plot_states, plot_nees, plot_states_al_in_one, plot_nees_al_in_one


@pytest.fixture
def config():
    with initialize(version_base=None, config_path="../configs"):
        cfg_generate_data = compose(config_name="generate_data", overrides=["local.base_dir=..", "N_timesteps_test=200"])
        cfg_fit_model = compose(config_name="fit_model", overrides=["local.base_dir=.."])
        cfg_run_filter = compose(config_name="run_filter", overrides=["local.base_dir=..", "N_timesteps_test=200"])
        combined_cfg = {'cfg_generate_data': cfg_generate_data, 'cfg_fit_model': cfg_fit_model, 'cfg_run_filter': cfg_run_filter}
    return combined_cfg


def test_run_pipeline(config):

    cfg_generate_data = config['cfg_generate_data']
    cfg_fit_model = config['cfg_fit_model']
    cfg_run_filter = config['cfg_run_filter']


    create_random_model(cfg_generate_data)
    generate_data(cfg_generate_data)
    fit_model(cfg_fit_model)

    run_filters(cfg_run_filter)

    # Show results
    # plot_measurement_and_prediction(cfg_run_filter.output_dir + '/kf_output.npz', 'Kalman Filter')
    # plot_measurement_and_prediction(cfg_run_filter.output_dir + '/rts_output.npz', 'RTS Smoother')

    # plot_states(cfg_run_filter.output_dir + '/kf_output.npz', 'Kalman Filter')
    # plot_states(cfg_run_filter.output_dir + '/rts_output.npz', 'RTS Smoother')

    plot_states_al_in_one(cfg_run_filter.output_dir + '/kf_output.npz', cfg_run_filter.output_dir + '/rts_output.npz')
    plot_nees_al_in_one(cfg_run_filter.output_dir + '/kf_output.npz', cfg_run_filter.output_dir + '/rts_output.npz')
    plt.show()

def test_run_filters(config):

    cfg_run_filter = config['cfg_run_filter']

    run_filters(cfg_run_filter)

    # Show results
    # plot_measurement_and_prediction(cfg_run_filter.output_dir + '/kf_output.npz', 'Kalman Filter')
    # plot_measurement_and_prediction(cfg_run_filter.output_dir + '/rts_output.npz', 'RTS Smoother')

    # plot_states(cfg_run_filter.output_dir + '/kf_output.npz', 'Kalman Filter')
    # plot_states(cfg_run_filter.output_dir + '/rts_output.npz', 'RTS Smoother')
    # plot_nees(cfg_run_filter.output_dir + '/kf_output.npz', 'Kalman Filter')
    # plot_nees(cfg_run_filter.output_dir + '/rts_output.npz', 'RTS Smoother')

    plot_states_al_in_one(cfg_run_filter.output_dir + '/kf_output.npz', cfg_run_filter.output_dir + '/rts_output.npz')
    plot_nees_al_in_one(cfg_run_filter.output_dir + '/kf_output.npz', cfg_run_filter.output_dir + '/rts_output.npz')

    plt.show()






        