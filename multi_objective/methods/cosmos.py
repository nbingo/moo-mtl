import torch
import torch.nn as nn
import numpy as np

from multi_objective.utils import num_parameters, calc_gradients, RunningMean
from .base import BaseMethod


class Upsampler(nn.Module):


    def __init__(self, K, child_model, input_dim, upsample_fraction=.75):
        """
        In case of tabular data: append the sampled rays to the data instances (no upsampling)
        In case of image data: use a transposed CNN for the sampled rays.
        """
        super().__init__()
        self.dim = input_dim
        self.fraction = upsample_fraction

        if len(input_dim) == 1:
            # tabular data
            self.tabular = True
        elif len(input_dim) == 3:
            # image data
            assert input_dim[-2] % 4 == 0 and input_dim[-1] % 4 == 0, 'Spatial image dim must be dividable by 4.'
            self.tabular = False
            self.transposed_cnn = nn.Sequential(
                nn.ConvTranspose2d(K, K, kernel_size=4, stride=1, padding=0, bias=False),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(K, K, kernel_size=6, stride=2, padding=1, bias=False),
            )
        else:
            raise ValueError(f"Unknown dataset structure, expected 1 or 3 dimensions, got {input_dim}")

        self.child_model = child_model


    def forward(self, batch):
        x = batch['data']
        
        b = x.shape[0]
        a = batch['alpha'].repeat(b, 1)

        if self.tabular:
             result = torch.cat((x, a), dim=1)
        else:
            # use transposed convolution
            a = a.reshape(b, len(batch['alpha']), 1, 1)
            a = self.transposed_cnn(a)

            target_size = (int(self.dim[-2]*self.fraction), int(self.dim[-1]*self.fraction))
            a = torch.nn.functional.interpolate(a, target_size, mode='nearest')
        
            # Random padding to avoid issues with subsequent batch norm layers.
            result = torch.randn(b, *self.dim, device=x.device)
            
            # Write x into result tensor
            channels = x.shape[1]
            result[:, 0:channels] = x

            # Write a into the middle of the tensor
            height_start = (result.shape[-2] - a.shape[-2]) // 2
            height_end = (result.shape[-2] - a.shape[-2]) - height_start
            width_start = (result.shape[-1] - a.shape[-1]) // 2
            width_end = (result.shape[-1] - a.shape[-1]) - width_start
            if height_start > 0:
                result[:, channels:, height_start:-height_end, width_start:-width_end] = a
            else:
                result[:, channels:] = a

        return self.child_model(dict(data=result))


class COSMOSMethod(BaseMethod):

    def __init__(self, objectives, model, cfg):
        """
        Instanciate the cosmos solver.

        Args:
            objectives: A list of objectives
            alpha: Dirichlet sampling parameter (list or float)
            lamda: Cosine similarity penalty
            dim: Dimensions of the data
            n_test_rays: The number of test rays used for evaluation.
        """
        super().__init__(objectives, model, cfg)
        self.K = len(objectives)
        self.alpha = cfg.cosmos.alpha
        self.lamda = cfg.cosmos.lamda

        if len(self.alpha) == 1:
            self.alpha = [self.alpha[0] for _ in self.task_ids]

        dim = list(cfg.dim)
        dim[0] = dim[0] + self.K

        self.data = RunningMean(20)     # should be updates per epoch
        self.alphas = RunningMean(20)

        model.change_input_dim(dim[0])
        self.model = Upsampler(self.K, model, dim).to(self.device)

        self.bn = {t: torch.nn.BatchNorm1d(1) for t in self.task_ids}

        self.n_params = num_parameters(self.model)
        print("Number of parameters: {}".format(self.n_params))


    def preference_at_inference(self):
        return True
    

    def new_epoch(self, e):
        if e > 2:
            data = np.array(self.data.queue)
            # self.means.append(data.mean(axis=0))
            # self.stds.append(data.std(axis=0) + 1e-8)

            # import matplotlib.pyplot as plt
            # fig, axes = plt.subplots(ncols=3)

            # axes[0].hist(data, bins=20, histtype='step')

            # trans = (data-data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
            # axes[1].hist(trans, bins=20, histtype='step')

            # sigm = 1/ (1 + np.exp(-trans))
            # axes[2].hist(sigm, bins=20, histtype='step')
            # plt.savefig(f'hist_{e}')
            # plt.close()


    def step(self, batch):
        # step 1: sample alphas
        
        alpha = np.random.dirichlet(self.alpha, 1).flatten()

        batch['alpha'] = torch.from_numpy(alpha.astype(np.float32)).to(self.device)

        # step 2: calculate loss
        self.model.zero_grad()
        logits = self.model(batch)
        batch.update(logits)
        loss_total = torch.tensor(0, device=self.device).float()
        task_losses = []
        task_losses_norm = []
        g_norms = []
        for i, (a, t) in enumerate(zip(batch['alpha'], self.task_ids)):
            task_loss = self.objectives[t](**batch)
            task_loss_norm = task_loss
            if len(self.data) > 2:
                # std = self.data.std(axis=0)[i]
                # mean = self.data(axis=0)[i]
                data = np.array(self.data.queue)
                min = data.min(axis=0)[i]
                max = data.max(axis=0)[i]
                # task_loss_norm = (task_loss - mean) / std    # z normalization (normalize scale)
                task_loss_norm = (task_loss - min) / (max - min + 1e-8)     # min max norm
                task_loss_norm = torch.abs(task_loss_norm)

                # scale to range of sampled alphas
                min_a = np.array(self.alphas.queue).min(axis=0)[i]
                max_a = np.array(self.alphas.queue).max(axis=0)[i]
                task_loss_norm = (task_loss_norm * (max_a - min_a)) + min_a
                # if task_loss_norm < -4:
                #     task_loss_norm *= -4 / task_loss_norm
                # if task_loss_norm > 4:
                #     task_loss_norm *= 4 / task_loss_norm
                # task_loss_norm = torch.sigmoid(task_loss_norm)
                # g_norms.append(std)
            loss_total += task_loss_norm * a
            task_losses.append(task_loss.item())
            task_losses_norm.append(task_loss_norm)
        
        # print(g_norms)
        # print([l.item() for l in task_losses_norm])
        # print([a.item() for a in batch['alpha']])
        # print(task_losses)
        
        loss_linearization = sum(task_losses)
        # loss_linearization = sum(task_losses_norm).item()

        cossim = torch.nn.functional.cosine_similarity(torch.stack(task_losses_norm), batch['alpha'], dim=0)
        loss_total -= self.lamda * cossim
        
        loss_total.backward()
        
        self.data.append(task_losses)
        self.alphas.append(batch['alpha'].cpu().detach().numpy())

        n = np.array([l.item() for l in task_losses_norm])

        return loss_linearization, cossim.item(), np.array2string(n, formatter={'float': '{: 0.3f}'.format})


    def eval_step(self, batch, preference_vector):
        self.model.eval()
        with torch.no_grad():
            batch['alpha'] = torch.from_numpy(preference_vector).to(self.device).float()
            return self.model(batch)