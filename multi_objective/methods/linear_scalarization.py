import torch
import numpy as np
from copy import deepcopy
from collections import OrderedDict
from .base import BaseMethod
from multi_objective import utils

class LinearScalarizationMethod(BaseMethod):

    def __init__(self, objectives, model, cfg):
        super().__init__(objectives, model, cfg)
        self.truncated_preference_rays = utils.reference_points(partitions=cfg['n_partitions'], dim=len(self.objectives))[1:]
        self.num_pref_rays = len(self.truncated_preference_rays)
        # The preference ray of the self.model model
        self.base_preference_ray = utils.reference_points(partitions=cfg['n_partitions'], dim=len(self.objectives))[0]
        if cfg.task_id is None:
            # Create copies for all reference points except for the first one, which will be handled by self.model
            self.models = [deepcopy(model) for _ in range(self.num_pref_rays)]
            self.optimizers = [torch.optim.Adam(m.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay) for m in self.models]
            self.schedulers = [utils.get_lr_scheduler(cfg.lr_scheduler, o, cfg, '') for o in self.optimizers]
            print('num models:', len(self.models) + 1)

        self.task_id = cfg.task_id
        self.preference_idx_at_inference = True
        

    # TODO: fix for dict
    def state_dict(self):
        if self.task_id is None:
            state = OrderedDict()
            for i, (m, o, s) in enumerate(zip(self.models, self.optimizers, self.schedulers)):
                state[f'model.{i}'] = m.state_dict()
                state[f'optimizer.{i}'] = o.state_dict()
                state[f'lr_scheduler.{i}'] = s.state_dict()
            return state

     # TODO: fix for dict
    def load_state_dict(self, dict):
        if self.task_id is None:
            for i in range(len(self.models)):
                self.models[i].load_state_dict(dict[f'model.{i}'])
                self.optimizers[i].load_state_dict(dict[f'optimizer.{i}'])
                self.schedulers[i].load_state_dict(dict[f'lr_scheduler.{i}'])



    def new_epoch(self, e):
        if self.task_id is None:
            for m in self.models:
                m.train()
            if e>0:
                for s in self.schedulers:
                    s.step()


    def step(self, batch):
        losses = []
        if self.task_id is None:
            for i, preference_ray in enumerate(self.truncated_preference_rays):
                optim = self.optimizers[i]
                model = self.models[i]
                optim.zero_grad()
                result = self._step(batch, model, preference_ray)
                optim.step()
                losses.append(result)
        
        # task zero we take the model we got via __init__
        self.model.zero_grad()
        result = self._step(batch, self.model, self.base_preference_ray)
        losses.append(result)
        return np.mean(losses).item()


    def _step(self, batch, model, preference_ray):
        batch.update(model(batch))
        # Take the weighted average of the objectives according to the preference ray
        loss = 0
        for i, objective in enumerate(self.objectives.values()):
            loss += objective(**batch) * preference_ray[i]
        loss.backward()
        return loss.item()


    def eval_step(self, batch, preference_vector, preference_vector_idx=None):
        with torch.no_grad():
            model = self.model if preference_vector_idx == 0 else self.models[preference_vector_idx - 1]
            return model(batch)

    def preference_at_inference(self):
        return True