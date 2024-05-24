import numpy as np
import matplotlib.pyplot as plt

def plot_measurement_and_prediction(npz_file):
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
    plt.title('z[0] for Measured and Predicted Values')
    plt.legend()
    plt.grid(True)
    # plt.show()

def plot_states(npz_file):
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
    plt.ylabel('z[0] Value')
    plt.title('z[0] for Measured and Predicted Values')
    plt.legend()
    plt.grid(True)
    # plt.show()