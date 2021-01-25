from abc import abstractmethod
import torch

from utils import model_from_dataset


class BaseSolver():


    def model_params(self):
        return list(self.model.parameters())

    
    def new_epoch(self, e):
        self.model.train()

    @abstractmethod
    def step(self, batch):
        raise NotImplementedError()
    

    def log(self):
        return {}

    @abstractmethod
    def eval_step(self, batch):
        raise NotImplementedError()
