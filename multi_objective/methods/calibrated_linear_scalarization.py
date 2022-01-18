import torch
import torch.nn as nn
import numpy as np

from multi_objective.utils import model_from_dataset
from .base import BaseMethod


class CalibratedLinearScalarizationMethod(BaseMethod):

    def __init__(self, objectives, model, cfg):
        super().__init__(objectives, model, cfg)
        self.alpha = cfg.alpha
        self.lamda = cfg.lamda
        self.K = len(objectives)

    
    def new_epoch(self, e):
        self.model.train()


    def step(self, batch):
        # step 1: sample alphas
        if isinstance(self.alpha, list):
            batch['alpha']  = torch.from_numpy(
                np.random.dirichlet(self.alpha, 1).astype(np.float32).flatten()
            ).cuda()
        elif self.alpha > 0:
            batch['alpha']  = torch.from_numpy(
                np.random.dirichlet([self.alpha for _ in range(self.K)], 1).astype(np.float32).flatten()
            ).cuda()
        else:
            raise ValueError(f"Unknown value for alpha: {self.alpha}, expecting list or float.")


        # step 2: calculate loss
        self.model.zero_grad()
        
        logits = self.model(batch)
        batch.update(logits)
        loss_total = None
        task_losses = []
        for a, t in zip(batch['alpha'], self.task_ids):
            task_loss = self.objectives[t](**batch)
            loss_total = a * task_loss if not loss_total else loss_total + a * task_loss
            task_losses.append(task_loss)
        cossim = torch.nn.functional.cosine_similarity(torch.stack(task_losses), batch['alpha'], dim=0)
        loss_total -= self.lamda * cossim
            
        loss_total.backward()
        return loss_total.item()


    def eval_step(self, batch, preference_vector):
        
        self.model.eval()
        with torch.no_grad():
            batch['alpha'] = torch.from_numpy(preference_vector).to(self.device).float()
            return self.model(batch)
        
    def preference_at_inference(self):
        return True