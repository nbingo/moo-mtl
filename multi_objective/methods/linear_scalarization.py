import torch

from multi_objective.utils import model_from_dataset
from .base import BaseMethod


class LinearScalarizationMethod(BaseMethod):

    def __init__(self, objectives, model, cfg):
        super().__init__(objectives, model, cfg)
        self.J = len(objectives)

    
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
        
        loss_total = torch.sum(task_losses)
        return loss_total.item()


    def eval_step(self, batch):
        self.model.eval()
        with torch.no_grad():
            return self.model(batch)