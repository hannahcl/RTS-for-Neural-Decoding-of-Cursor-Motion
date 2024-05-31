import numpy as np
import matplotlib.pyplot as plt

from src.analysis.nees import compute_nees_bounds

def plot_measurement_and_prediction(npz_file: str, title: str = ''):
    # Load the data from the .npz file
    data = np.load(npz_file)

    # Extract the arrays
    timesteps = data['timesteps']
    z_mes_vals = data['z_mes_vals']
    z_pred_vals = data['z_pred_vals']

    # Plot z[0] for z_mes and z_pred
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, z_mes_vals[:, 0], label='z_mes[0]')
    plt.plot(timesteps, z_pred_vals[:, 0], label='z_pred[0]')
    plt.xlabel('Timesteps')
    plt.ylabel('z[0] Value')
    plt.title(title + ': z[0] for Measured and Predicted Values')
    plt.legend()
    plt.grid(True)
    # plt.show()

def plot_states(npz_file: str, title: str = ''):
    # Load the data from the .npz file
    data = np.load(npz_file)

    # Extract the arrays
    timesteps = data['timesteps']
    x_gt_vals = data['x_gt_vals']
    x_est_vals = data['x_est_vals']

    # Plot z[0] for z_mes and z_pred
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, x_gt_vals[:, 0], label='x pos gt')
    plt.plot(timesteps, x_est_vals[:, 0], label='x pos estimate')
    plt.xlabel('Timesteps')
    plt.ylabel('state Value')
    plt.title(title + 'Pos x for Ground Truth and Estimate')
    plt.legend()
    plt.grid(True)
    # plt.show()


def plot_states_al_in_one(npz_kf, npz_rts):
    # Load the data from the .npz file
    data_kf = np.load(npz_kf)
    data_rts = np.load(npz_rts)

    # Extract the arrays
    timesteps = data_kf['timesteps']
    x_gt_vals = data_kf['x_gt_vals']

    x_est_kf = data_kf['x_est_vals']
    x_est_rts = data_rts['x_est_vals']
    x_est_smooth = data_rts['x_est_smooth_vals']

    # Plot z[0] for z_mes and z_pred
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, x_gt_vals[:, 0], label='gt')
    plt.plot(timesteps, x_est_kf[:, 0], label='KF')
    plt.plot(timesteps, x_est_rts[:, 0], label='RTS')
    plt.plot(timesteps, x_est_smooth[:, 0], label='Smooth')
    plt.xlabel('Timesteps')
    plt.ylabel('state Value')
    plt.title('Position in x-direction from ground truth, KF, RTS, and RTS smoothed.')
    plt.legend()
    plt.grid(True)

def plot_nees_al_in_one(npz_kf, npz_rts):
    # Load the data from the .npz file
    data_kf = np.load(npz_kf)
    data_rts = np.load(npz_rts)

    # Extract the arrays
    timesteps = data_kf['timesteps']
    nees_kf = data_kf['nees_vals']
    nees_rts = data_rts['nees_vals']
    nees_smoothed = data_rts['nees_smooth_vals']

    lb, ub = compute_nees_bounds()


    # Plot z[0] for z_mes and z_pred
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, nees_kf, label='KF')
    plt.plot(timesteps, nees_rts, label='RTS')
    plt.plot(timesteps, nees_smoothed, label='Smoothed')
    plt.axhline(y=lb, color='r', linestyle='--', label=f'Lower Bound ')
    plt.axhline(y=ub, color='g', linestyle='--', label=f'Upper Bound ')
    plt.xlabel('Timesteps')
    plt.ylabel('state Value')
    plt.title('Normalized Estimation Error Squared')
    plt.legend()
    plt.grid(True)



def plot_nees(npz_file: str, title: str = ''):
    # Load the data from the .npz file
    data = np.load(npz_file)

    # Extract the arrays
    timesteps = data['timesteps']
    nees = data['nees_vals']

    lb, ub = compute_nees_bounds()


    # Plot z[0] for z_mes and z_pred
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, nees, label='NEES')
    plt.axhline(y=lb, color='r', linestyle='--', label=f'Lower Bound ')
    plt.axhline(y=ub, color='g', linestyle='--', label=f'Upper Bound ')
    plt.xlabel('Timesteps')
    plt.ylabel('state Value')
    plt.title(title + 'Normalized Estimation Error Squared')
    plt.legend()
    plt.grid(True)