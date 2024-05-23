import hydra
from omegaconf import DictConfig, OmegaConf
import random
import numpy as np
import pickle
from pathlib import Path
import os

from src.models.gt_model import EstimatedMesurmentModel



@hydra.main(version_base=None, config_path="../configs", config_name="fit_model")
def fit_model(cfg : DictConfig) -> None:
    # print(OmegaConf.to_yaml(cfg))


    model = EstimatedMesurmentModel(cfg)
    model.fit(cfg.input_dir + 'train_data.pkl')

    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)

    assert model.H is not None
    np.savez(cfg.output_dir + '/model_H.npz', H=model.H)

if __name__ == "__main__":
    fit_model()