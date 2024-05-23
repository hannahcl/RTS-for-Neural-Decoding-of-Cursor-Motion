from nptyping import NDArray, Shape, Float
import numpy as np
from omegaconf import DictConfig
import random
import pickle



class GTProcessModel:

    def __init__(self, cfg: DictConfig) -> None:
        self.A = cfg.gt_model.A
        self.q_vec = np.array(cfg.gt_model.q_vec)

        self.nx = cfg.gt_model.nx
        self.dt = cfg.timestep

        assert self.q_vec.shape == (self.nx,)

    def step_once(self, x: NDArray[Shape["6"], Float]) -> NDArray[Shape["2, 2"], Float]:
        dx = self.A@x + np.random.normal(np.zeros(self.nx), self.q_vec)
        return x + dx*self.dt


class GTMeasurmentModel:

    def __init__(self, cfg: DictConfig) -> None:

        self.r = cfg.gt_model.r
        self.C = cfg.gt_model.C

        self.H1 = np.array(cfg.gt_model.H1)
        self.H2 = np.array(cfg.gt_model.H2)

    def get_measurement_nonlin(self, x: NDArray[Shape["6"], Float]) -> NDArray[Shape["42"], Float]:
        x = x.reshape((-1,1))
        order1 = self.H1@x
        order2 = 0.001*self.H2@x@x.T@self.H2.T
        z = order1 + order2@order1 + np.random.normal(np.zeros(self.C), np.ones(self.C)*self.r).reshape((-1,1))
        return z.reshape((-1,))
    
    def get_measurement_lin(self, x: NDArray[Shape["6"], Float]) -> NDArray[Shape["42"], Float]:
        x = x.reshape((-1,1))
        z = self.H1@x + np.random.normal(np.zeros(self.C), np.ones(self.C)*self.r).reshape((-1,1))
        return z.reshape((-1,))
    
class EstimatedMesurmentModel:

    def __init__(self, cfg: DictConfig) -> None:
        self.r : float = cfg.gt_model.r
        self.C : float = cfg.gt_model.C

        self.H : NDArray[Shape["42,6"], Float] | None = None

    def fit(self, data_file) -> None:

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
        self.H = H.T

    def set_model(self, H: NDArray[Shape["42, 6"], Float]) -> None:
        self.H = H

    def get_measurement(self, x: NDArray[Shape["6"], Float]) -> NDArray[Shape["42"], Float]:
        x = x.reshape((-1,1))
        z = self.H@x + np.random.normal(np.zeros(self.C), np.ones(self.C)*self.r).reshape((-1,1))

        return z.reshape((-1,))
        
