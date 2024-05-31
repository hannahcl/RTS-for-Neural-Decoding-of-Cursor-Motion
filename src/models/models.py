from nptyping import NDArray, Shape, Float
from typing import Tuple, Optional
import numpy as np
from omegaconf import DictConfig
import pickle


class ProcessModel:

    def __init__(self, cfg: DictConfig) -> None:
        self.A = np.array(cfg.model.A)
        self.q_vec = np.array(cfg.model.q_vec)

        self.nx = cfg.model.nx
        self.dt = cfg.timestep

        assert self.q_vec.shape == (self.nx,)
        assert self.A.shape == (self.nx, self.nx)

    def step_once(self, x: NDArray[Shape["6"], Float]) -> NDArray[Shape["6, 6"], Float]:
        dx = self.A@x + np.random.normal(np.zeros(self.nx), self.q_vec)
        return x + dx*self.dt


class MeasurmentModel:

    def __init__(self, cfg: DictConfig) -> None:

        self.cfg = cfg
        self.r = cfg.model.r
        self.C = cfg.model.C

        self.H1: NDArray[Shape["42, 6"], Float] | None = None
        self.H2: NDArray[Shape["42, 6"], Float] | None = None

        self._load_model()
    
    def _load_model(self) -> None:
        assert self.cfg.path_to_measurement_model is not None, "Model is not provided"
        print(f'Loading model from {self.cfg.path_to_measurement_model}')
        data = np.load(self.cfg.path_to_measurement_model)
        self.H1 = data['H1']

    def get_model(self) -> Tuple[NDArray[Shape["42, 6"], Float], Optional[NDArray[Shape["42, 6"], Float]]]:
        return self.H1, self.H2
    
    def get_measurement(self, x: NDArray[Shape["6"], Float]) -> NDArray[Shape["42"], Float]:
        x = x.reshape((-1,1))
        z = self.H1@x
        z = z + np.random.normal(np.zeros(self.C), np.ones(self.C)*self.r).reshape((-1,1))
        return z.reshape((-1,))
        
      
