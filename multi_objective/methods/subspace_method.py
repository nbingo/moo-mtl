import torch

from multi_objective.utils import model_from_dataset
from .base import BaseMethod


class SubspaceMethod(BaseMethod):

    def __init__(self, objectives, model, cfg):
        super().__init__(objectives, model, cfg)
        self.K = len(objectives)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.scheduler = utils.get_lr_scheduler(cfg.scheduler, self.optimizer, '')

    def state_dict(self):
        state_dict = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.scheduler.state_dict()
        }
        return state_dict()
    
    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)
        self.optimizer.load_state_dict(state_dict)
        self.scheduler.load_state_dict(state_dict)
    
    
    def new_epoch(self, e):
        self.model.train()
        if e > 0:
            self.scheduler.step()

    def step(self, batch):
        # step 1: sample alphas
        if isinstance(self.alpha, list):
            alpha  = torch.from_numpy(
                np.random.dirichlet(self.alpha, 1).astype(np.float32).flatten()
            ).cuda()
        elif self.alpha > 0:
            alpha  = torch.from_numpy(
                np.random.dirichlet([self.alpha for _ in range(self.K)], 1).astype(np.float32).flatten()
            ).cuda()
        else:
            raise ValueError(f"Unknown value for alpha: {self.alpha}, expecting list or float.")


        # step 2: calculate loss
        self.model.zero_grad()
        
        logits = self.model(batch, alpha)
        batch.update(logits)
        loss_total = None
        task_losses = []
        for a, t in zip(alpha, self.task_ids):
            task_loss = self.objectives[t](**batch)
            loss_total = a * task_loss if not loss_total else loss_total + a * task_loss
            task_losses.append(task_loss)
        
        loss_total.backward()
        return loss_total.item()



    def eval_step(self, batch, preference_vector):
        self.model.eval()
        with torch.no_grad():
            return self.model(batch, alpha=preference_vector)
        
    def preference_at_inference(self):
        return True