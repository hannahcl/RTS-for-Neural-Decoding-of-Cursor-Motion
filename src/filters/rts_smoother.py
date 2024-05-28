import numpy as np
from nptyping import NDArray, Shape, Float
from typing import Tuple

from src.filters.kalman_filter import KF

class RTSSmoother:
    def __init__(self, cfg: dict) -> None:
    
        self.lag = 5

        self.x_after_forward = np.zeros((self.lag, 6))
        self.P_after_forward = np.zeros((self.lag, 6, 6))
        self.x_after_backward = np.zeros((self.lag, 6))
        self.P_after_backward = np.zeros((self.lag, 6, 6))

        self.kf = KF(cfg)
        self.kf_backward = BackwardKF(cfg)
        xi = np.array(cfg.xi)

        self.mes = np.array([self.kf.H@xi for _ in range(self.lag)])
        self.x_after_backward[-1] = xi
        self.P_after_backward[-1] = np.array(cfg.Pi)

    def update(self, z: NDArray[Shape['42'], Float]) -> None:
        self._set_new_mes(z)
        self._forward_pass()
        self._backward_pass()
    
    def get_oldest_estimate(self) -> Tuple[NDArray[Shape['6'], Float], NDArray[Shape['6,6'], Float]]:
        return self.x_after_backward[0], self.P_after_backward[0]
    
    def get_newest_estimate(self) -> Tuple[NDArray[Shape['6'], Float], NDArray[Shape['6,6'], Float]]:
        return self.x_after_forward[-1], self.P_after_forward[-1]

    def _set_new_mes(self, z: NDArray[Shape['42'], Float]) -> None:
        self.mes = np.roll(self.mes, -1, axis=0)
        self.mes[-1] = z

    def _forward_pass(self) -> None:
        self.x_after_forward[0], self.P_after_forward[0] = self.kf.update(
            self.x_after_backward[0],
            self.P_after_backward[0],
            self.mes[0]
        )
        for i in range(1, self.lag):
            self.x_after_forward[i], self.P_after_forward[i] = self.kf.update(
                self.x_after_forward[i-1],
                self.P_after_forward[i-1],
                self.mes[i]
            )

    def _backward_pass(self) -> None:
        self.x_after_backward[-1] = self.x_after_forward[-1]
        self.P_after_backward[-1] = self.P_after_forward[-1]
        for i in range(self.lag-2, -1, -1):
            self.x_after_backward[i], self.P_after_backward[i] = self.kf_backward.update(
                self.x_after_forward[i],
                self.P_after_forward[i],
                self.x_after_backward[i+1],
                self.P_after_backward[i+1]
            )       
    
class BackwardKF:
    def __init__(self, cfg: dict) -> None:
        self.kf = KF(cfg)
        self.A = self.kf.A
        self.Q = self.kf.Q
        self.H = self.kf.H
        self.R = self.kf.R

    def update(
        self,
        xk_forward: NDArray[Shape['6'], Float],
        Pk_forward: NDArray[Shape['6,6'], Float],
        xk1_smoothed: NDArray[Shape['6'], Float],
        Pk1_smoothed: NDArray[Shape['6,6'], Float]
        ) -> Tuple[
            NDArray[Shape['6'], Float],
            NDArray[Shape['6,6'], Float]]:
        
        Pk1_forward = self.kf.A@Pk_forward@self.kf.A.T + self.kf.Q
        W = Pk_forward@self.kf.A.T@np.linalg.inv(Pk1_forward)

        xk_after_backward = xk_forward + W@(xk1_smoothed - self.kf.A@xk_forward)
        Pk_after_backward = Pk_forward + W@(Pk1_smoothed - Pk1_forward)@W.T

        return xk_after_backward, Pk_after_backward
