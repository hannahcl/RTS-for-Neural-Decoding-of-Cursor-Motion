import numpy as np
from nptyping import NDArray, Shape, Float
from typing import Tuple


from src.models.models import ProcessModel, MeasurmentModel

class KF:

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg

        self.A = None
        self.Q = None
        self.H = None
        self.R = None

        self._get_discrete_model()        

    def _get_discrete_model(self) -> None:
        model = ProcessModel(self.cfg)
        mes_model = MeasurmentModel(self.cfg)

        dt = self.cfg.timestep

        self.A = np.eye(6) + model.A*dt
        
        self.Q = np.diag(np.array(model.q_vec)) #TODO discretize

        self.H = mes_model.get_model()[0]
        self.R = mes_model.r*np.eye(self.H.shape[0])
    
    def update(
            self,
            x: NDArray[Shape["6"], Float],
            P: NDArray[Shape["6,6"], Float],
            z: NDArray[Shape["42"], Float]
            ) -> Tuple[
                NDArray[Shape["6"], Float],
                NDArray[Shape["6,6"], Float]]:
        
        x_predicted = self.A@x
        P_predicted = self.A@P@self.A.T + self.Q

        z_predicted = self.H@x_predicted
        S_predicted = self.H@P_predicted@self.H.T + self.R

        K = P_predicted@self.H.T@np.linalg.inv(S_predicted)

        x_posteriori = x_predicted - K@(z_predicted- z)
        P_posteriori = (np.eye(6) - K@self.H)@P_predicted@(np.eye(6) - K@self.H).T + K@self.R@K.T

        return (x_posteriori, P_posteriori)