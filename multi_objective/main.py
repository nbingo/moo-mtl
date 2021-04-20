import torch
import random
import numpy as np

# seed now to be save and overwrite later
np.random.seed(1)
random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.cuda.manual_seed_all(1)

import argparse
import logging
import os
import pathlib
import time
import json
import math
import matplotlib.pyplot as plt
from torch.utils import data
from fvcore.common.config import CfgNode

from rtb import log_every_n_seconds, log_first_n, setup_logger

import defaults
import utils
from objectives import from_name


from methods import HypernetMethod, ParetoMTLMethod, SingleTaskMethod, COSMOSMethod, MGDAMethod, UniformScalingMethod
from scores import from_objectives


def method_from_name(method, objectives, model, cfg):
    """
    Initializes the method specified in settings along with its configuration.

    Args:
        objectives (dict): All objectives for the experiment. Structure is
            task_id: Objective.
        model (models.base.BaseModel): A model for the method to learn. It's a
            `torch.nn.Module` with some custom functions required by some MOO methods
        settings (dict): The settings
    
    Returns:
        Method. The configured method instance.
    """
    if method == 'pmtl':
        return ParetoMTLMethod(objectives, model, cfg)
    elif 'cosmos' in method:
        return COSMOSMethod(objectives, model, cfg)
    elif method == 'single_task':
        return SingleTaskMethod(objectives, model, cfg)
    elif 'phn' in method:
        return HypernetMethod(objectives, model, cfg)
    elif method == 'mgda':
        return MGDAMethod(objectives, model, cfg)
    elif method == 'uniform':
        return UniformScalingMethod(objectives, model, **cfg)
    else:
        raise ValueError("Unkown method {}".format(method))


def evaluate(j, e, method, scores, data_loader, split, result_dict, logdir, train_time, cfg, logger):
    """
    Evaluate the method on a given dataset split. Calculates:
    - score for all the scores given in `scores`
    - computes hyper-volume if applicable
    - plots the Pareto front to `logdir` for debugging purposes

    Also stores everything in a json file.

    Args:
        j (int): The index of the run (if there are several starts)
        e (int): Epoch
        method: The method subject to evaluation
        scores (dict): All scores which the method should be evaluated on
        data_loader: The dataloader
        split (str): Split of the evaluation. Used to name log files
        result_dict (dict): Global result dict to store the evaluations for this epoch and run in
        logdir (str): Directory where to store the logs
        train_time (float): The training time elapsed so far, added to the logs
        settings (dict): Settings of the experiment
    
    Returns:
        dict: The updates `result_dict` containing the results of this evaluation
    """
    assert split in ['train', 'val', 'test']
    
    if len(cfg.task_ids) > 0:
        J = len(cfg['task_ids'])
        task_ids = cfg['task_ids']
    else:
        # single output setting
        J = len(cfg['objectives'])
        task_ids = list(scores[list(scores)[0]].keys())

    pareto_rays = utils.reference_points(cfg['n_partitions'], dim=J)
    n_rays = pareto_rays.shape[0]
    
    log_first_n(logging.DEBUG, f"Number of test rays: {n_rays}", n=1)
    
    # gather the scores
    score_values = {et: utils.EvalResult(J, n_rays, task_ids) for et in scores.keys()}
    for b, batch in enumerate(data_loader):
        log_every_n_seconds(logging.INFO, f"Eval {b} of {len(data_loader)}", n=5)
        batch = utils.dict_to(batch, cfg['device'])
                
        if method.preference_at_inference():
            data = {et: np.zeros((n_rays, J)) for et in scores.keys()}
            for i, ray in enumerate(pareto_rays):
                logits = method.eval_step(batch, preference_vector=ray)
                batch.update(logits)

                for eval_mode, score in scores.items():

                    data[eval_mode][i] += np.array([score[t](**batch) for t in task_ids])
            
            for eval_mode in scores:
                score_values[eval_mode].update(data[eval_mode], 'pareto_front')
        else:
            # Method gives just a single point
            batch.update(method.eval_step(batch))
            for eval_mode, score in scores.items():
                data = [score[t](**batch) for t in task_ids]
                score_values[eval_mode].update(data, 'single_point')


    # normalize scores and compute hyper-volume
    for v in score_values.values():
        v.normalize()
        if method.preference_at_inference():
            v.compute_hv(cfg['reference_point'])
            v.compute_optimal_sol()

    # plot pareto front to pf
    for eval_mode, score in score_values.items():
        pareto_front = utils.ParetoFront(
            ["-".join([str(t), eval_mode]) for t in task_ids], 
            logdir,
            "{}_{}_{:03d}".format(eval_mode, split, e)
        )
        if score.pf_available:
            pareto_front.plot(score.pf, best_sol_idx=score.optimal_sol_idx)
        else:
            pareto_front.plot(score.center)

    result = {k: v.to_dict() for k, v in score_values.items()}
    result.update({"training_time_so_far": train_time,})
    result.update(method.log())

    if f"epoch_{e}" in result_dict[f"start_{j}"]:
        result_dict[f"start_{j}"][f"epoch_{e}"].update(result)
    else:
        result_dict[f"start_{j}"][f"epoch_{e}"] = result

    with open(pathlib.Path(logdir) / f"{split}_results.json", "w") as file:
        json.dump(result_dict, file)
    
    return result_dict


def main(method_name, cfg, tag=''):
    # create the experiment folders
    logdir = os.path.join(cfg['logdir'], method_name, cfg['dataset'], utils.get_runname(cfg) + f'_{tag}')
    pathlib.Path(logdir).mkdir(parents=True, exist_ok=True)

    logger = setup_logger(os.path.join(logdir, 'exp.log'), name=__name__)
    logger.info(f"start experiment with settings {cfg}")

    # prepare
    utils.set_seed(cfg['seed'])
    lr_scheduler = cfg[method_name].lr_scheduler

    objectives = from_name(**cfg)
    scores = from_objectives(objectives, **cfg)

    train_loader, val_loader, test_loader = utils.loaders_from_name(**cfg)

    rm1 = utils.RunningMean(len(train_loader))
    rm2 = utils.RunningMean(len(train_loader))
    elapsed_time = 0

    model = utils.model_from_dataset(**cfg).to(cfg.device)
    method = method_from_name(method_name, objectives, model, cfg)

    utils.GradientMonitor.register_parameters(model, filter='encoder')

    train_results = dict(settings=cfg, num_parameters=utils.num_parameters(method.model_params()))
    val_results = dict(settings=cfg, num_parameters=utils.num_parameters(method.model_params()))
    test_results = dict(settings=cfg, num_parameters=utils.num_parameters(method.model_params()))

    with open(pathlib.Path(logdir) / "settings.json", "w") as file:
        json.dump(train_results, file)

    # main
    num_starts = cfg[method_name].num_starts if 'num_starts' in cfg[method_name] else 1
    for j in range(num_starts):
        train_results[f"start_{j}"] = {}
        val_results[f"start_{j}"] = {}
        test_results[f"start_{j}"] = {}

        optimizer = torch.optim.Adam(method.model_params(), cfg.lr)

        if lr_scheduler == 'None':
            scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda epoch: 1.)    # does nothing to the lr
        elif lr_scheduler == "CosineAnnealing":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'])
        elif lr_scheduler == "MultiStep":
            # if cfg['scheduler_milestones'] is None:
            milestones = [int(.33 * cfg['epochs']), int(.66 * cfg['epochs'])]
            # else:
            #     milestones = cfg['scheduler_milestones']
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)
        else:
            raise ValueError(f"Unknown lr scheduler {lr_scheduler}")
        
        for e in range(cfg['epochs']):
            tick = time.time()
            method.new_epoch(e)

            for b, batch in enumerate(train_loader):
                batch = utils.dict_to(batch, cfg.device)
                optimizer.zero_grad()
                stats = method.step(batch)
                
                # utils.GradientMonitor.collect_grads(model)
                
                optimizer.step()

                loss, sim, norm  = stats if isinstance(stats, tuple) else (stats, 0, 0)
                assert not math.isnan(loss) and not math.isnan(sim)
                log_every_n_seconds(logging.INFO, 
                    f"Epoch {e:03d}, batch {b:03d}, train_loss {loss:.4f}, rm train_loss {rm1(loss):.3f}, rm sim {rm2(sim):.3f}",
                    n=5
                )

            tock = time.time()
            elapsed_time += (tock - tick)

            
            val_results[f"start_{j}"][f"epoch_{e}"] = {'lr': scheduler.get_last_lr()[0]}
            scheduler.step()

            # run eval on train set (mainly for debugging)
            if cfg['train_eval_every'] > 0 and (e+1) % cfg['train_eval_every'] == 0:
                train_results = evaluate(j, e, method, scores, train_loader,
                    split='train',
                    result_dict=train_results,
                    logdir=logdir,
                    train_time=elapsed_time,
                    cfg=cfg,
                    logger=logger,)

            
            if cfg['eval_every'] > 0 and (e+1) % cfg['eval_every'] == 0:
                # Validation results
                val_results = evaluate(j, e, method, scores, val_loader,
                    split='val',
                    result_dict=val_results,
                    logdir=logdir,
                    train_time=elapsed_time,
                    cfg=cfg,
                    logger=logger,)

                # Test results
                # test_results = evaluate(j, e, method, scores, test_loader,
                #     split='test',
                #     result_dict=test_results,
                #     logdir=logdir,
                #     train_time=elapsed_time,
                #     settings=settings,)

            # Checkpoints
            if cfg['checkpoint_every'] > 0 and (e+1) % cfg['checkpoint_every'] == 0:
                pathlib.Path(os.path.join(logdir, 'checkpoints')).mkdir(parents=True, exist_ok=True)
                torch.save(method.model.state_dict(), os.path.join(logdir, 'checkpoints', 'c_{}-{:03d}.pth'.format(j, e)))

        pathlib.Path(os.path.join(logdir, 'checkpoints')).mkdir(parents=True, exist_ok=True)
        torch.save(method.model.state_dict(), os.path.join(logdir, 'checkpoints', 'c_{}-{:03d}.pth'.format(j, 999999)))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help="Config file.")
    parser.add_argument('--method', '-m', default='cosmos', help="The method to generate the Pareto front.")
    parser.add_argument('--tag', default='', type=str, help="Experiment tag")
    parser.add_argument('--seed', '-s', default=1, type=int, help="Seed")
    parser.add_argument('--task_id', '-t', default=None, help='Task id to run single task in parallel. If not set then sequentially.')
    args = parser.parse_args()

    cfg = defaults.get_cfg_defaults()
    cfg.merge_from_file(args.config)
    
    if args.method == 'single_task' and args.task_id is not None:
            cfg.single_task.task_id = args.task_id
    
    cfg.seed = args.seed
    cfg.freeze()

    tag = args.tag
    if args.method == 'single_task':
        tag += str(cfg.single_task.task_id)

    return args.method, cfg, tag


if __name__ == "__main__":
    
    method, cfg, tag = parse_args()
    main(method, cfg, tag)
