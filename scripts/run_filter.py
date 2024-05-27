import hydra
from omegaconf import DictConfig, OmegaConf
import random
import numpy as np
import pickle
from pathlib import Path
import os

from src.filters.kalman_filter import KF
from src.filters.rts_smoother import RTSSmoother

def run_kf_and_store(cfg: dict) -> None:
    kf = KF(cfg)

    # Run filter on test data
    num_timesteps = cfg.N_timesteps_test
    timesteps = np.zeros(num_timesteps, dtype=int)
    x_gt_vals = np.zeros((num_timesteps, cfg.model.nx))
    x_est_vals = np.zeros((num_timesteps, cfg.model.nx))
    P_est_vals = np.zeros((num_timesteps, cfg.model.nx, cfg.model.nx))
    z_mes_vals = np.zeros((num_timesteps, cfg.model.C))

    x_est = np.array(cfg.xi)
    P_est = np.array(cfg.Pi)
    with open(cfg.input_dir + 'test_data.pkl', 'rb') as file:
        idx = 0
        try:
            while True:
                k, x_gt, z = pickle.load(file)
                x_est, P_est = kf.update(x_est, P_est, z)

                timesteps[idx] = k
                x_gt_vals[idx] = x_gt
                x_est_vals[idx] = x_est
                P_est_vals[idx] = P_est
                z_mes_vals[idx] = z

                idx += 1
        except EOFError:
            pass  # End of file reached

    # Store the results
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)
    output_file = os.path.join(cfg.output_dir, 'kf_output.npz')
    np.savez(output_file, 
             timesteps=timesteps, 
             x_gt_vals=x_gt_vals, 
             x_est_vals=x_est_vals, 
             z_mes_vals=z_mes_vals,)
    
def run_rts_and_store(cfg: dict) -> None:
    rts = RTSSmoother(cfg)

    # Run filter on test data
    num_timesteps = cfg.N_timesteps_test
    timesteps = np.zeros(num_timesteps, dtype=int)
    x_gt_vals = np.zeros((num_timesteps, cfg.model.nx))
    x_est_vals = np.zeros((num_timesteps, cfg.model.nx))
    P_est_vals = np.zeros((num_timesteps, cfg.model.nx, cfg.model.nx))
    z_mes_vals = np.zeros((num_timesteps, cfg.model.C))

    x_est = np.array(cfg.xi)
    P_est = np.array(cfg.Pi)
    with open(cfg.input_dir + 'test_data.pkl', 'rb') as file:
        idx = 0
        try:
            while True:
                k, x_gt, z = pickle.load(file)
                rts.update(z)
                x_est, P_est = rts.get_oldest_estimate()

                timesteps[idx] = k
                x_gt_vals[idx] = x_gt
                x_est_vals[idx] = x_est
                P_est_vals[idx] = P_est
                z_mes_vals[idx] = z

                idx += 1
        except EOFError:
            pass  # End of file reached

    # Store the results
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)
    output_file = os.path.join(cfg.output_dir, 'rts_output.npz')
    np.savez(output_file, 
             timesteps=timesteps, 
             x_gt_vals=x_gt_vals, 
             x_est_vals=x_est_vals, 
             z_mes_vals=z_mes_vals,)

@hydra.main(version_base=None, config_path="../configs", config_name="run_filter")
def run_filters(cfg : DictConfig) -> None:

    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)

    random.seed(cfg.seed)

    run_kf_and_store(cfg)
    run_rts_and_store(cfg)



if __name__ == "__main__":
    run_filters()