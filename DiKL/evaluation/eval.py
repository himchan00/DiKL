import torch
import torch.optim as optim
from torch.distributions import Normal, Gumbel
import numpy as np
import random

import os
import yaml
import argparse
from tqdm import tqdm
from types import SimpleNamespace as config

from train_utils import get_network, get_target, get_sample, save_plot_and_check, total_variation_distance
from models.dm_utils import extract, remove_mean
from loss import dsm_loss, diKL_loss


def parse_args():
    parser = argparse.ArgumentParser(description="Configuration for model training")

    parser.add_argument("--target", type=str, default='mog')
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--model_dir", type=str)
    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    if args.target == 'mog':
        cfg = 'configs/mog.yaml'
    elif args.target == 'mw':
        cfg = 'configs/mw.yaml'
    elif args.target == 'dw':
        cfg = 'configs/dw.yaml'
    elif args.target == 'lj':
        cfg = 'configs/lj.yaml'
    else:
        raise NotImplementedError
    with open(cfg, 'r') as file:
        opt = config(**yaml.safe_load(file))
    opt.ais = config(**opt.ais)

    print("target:", args.target)
    
    # make dir to save results
    opt.device = args.device
    opt.model_dir = args.model_dir


    target = get_target(opt)

    lvm, score_model = get_network(opt)
    lvm_optim = optim.Adam(lvm.parameters(), lr=opt.lvm_lr)
    score_optim = optim.Adam(score_model.parameters(), lr=opt.score_lr)

    process = (lambda x: remove_mean(x, n_particles=opt.n_particles, n_dimensions=opt.n_dim)) if opt.e3 else (lambda x: x)

    best_metric = 100
    for it in tqdm(range(1,opt.max_iter+1)):
        # train score network
        lvm.eval()
        score_model.train()
        Score_loss = []
        for _ in range(opt.score_iter):
            score_optim.zero_grad()
            x = get_sample(lvm, opt, stop_grad=True)
            # dsm loss
            score_loss = dsm_loss(score_model, x, opt)
            score_loss.backward()
            score_optim.step()
            Score_loss.append(score_loss.item())
        
        # train LVM with diffusive rKL
        lvm.train()
        score_model.eval()
        lvm_optim.zero_grad()
        x = get_sample(lvm, opt, stop_grad=False)

        lvm_loss, posterior_samples = diKL_loss(score_model, x, process, opt, target)
        if ~(torch.isnan(lvm_loss) | torch.isinf(lvm_loss)):
            lvm_loss.backward()

        torch.nn.utils.clip_grad_norm_(lvm.parameters(), opt.grad_norm_clip)
        lvm_optim.step()

        # plot and save checkpoints
        if it % opt.check_iter == 0 or it == 1:
            lvm.eval()
            score_model.eval()
            x_samples = get_sample(lvm, opt, True, opt.eval_samples)
            metric = save_plot_and_check(opt, x_samples, posterior_samples, target, plot_file_name=opt.proj_path + '/plot/%d.png'%it)
            if metric <= best_metric:
                best_metric = metric
                # save ckpt
                torch.save(lvm.state_dict(), opt.proj_path + '/model/' + 'LVM.pt')
                torch.save(score_model.state_dict(), opt.proj_path + '/model/' + 'SCORE.pt')
                if opt.early_stop:
                    print('Iter %d, '%it, 'Metric %.6f'%metric, flush=True)
            if args.track_tvd:
                val_data = torch.from_numpy(np.load(opt.val_sample_path))
                def total_variation_distance(samples1, samples2, bins=200):
                    min_ = 0.0
                    max_ = 8.0
                    # Create histograms of the two sample sets
                    hist1, bins = np.histogram(samples1, bins=bins, range=(min_, max_))
                    hist2, _ = np.histogram(samples2, bins=bins, range=(min_, max_))

                    if sum(hist1) / samples1.shape[0] < 0.6: #  in case that the samples are outside [min, max]
                        return 1e10
                    
                    # Normalize histograms to get probability distributions
                    hist1 = hist1 / np.sum(hist1)
                    hist2 = hist2 / np.sum(hist2)
                    
                    # Calculate the Total Variation distance
                    tv_distance = 0.5 * np.sum(np.abs(hist1 - hist2))
                    
                    return tv_distance

                x = (((val_data.reshape(-1, opt.n_particles, 1, opt.n_dim) - val_data.reshape(-1, 1, opt.n_particles, opt.n_dim))**2).sum(-1).sqrt()).cpu()
                diagx = torch.triu_indices(x.shape[1], x.shape[1], 1)
                val_data_dist = x[:, diagx[0], diagx[1]].flatten()
                 
                x = (((x_samples.reshape(-1, opt.n_particles, 1, opt.n_dim) - x_samples.reshape(-1, 1, opt.n_particles, opt.n_dim))**2).sum(-1).sqrt()).cpu()
                diagx = torch.triu_indices(x.shape[1], x.shape[1], 1)
                last_samples_dist = x[:, diagx[0], diagx[1]].flatten()

                tvd = total_variation_distance(
                                                val_data_dist.detach().cpu().numpy(), #target.energy(val_data).detach().cpu().numpy(), 
                                                last_samples_dist.detach().cpu().numpy(), #target.energy(x_samples).detach().cpu().numpy(), 
                                                bins=200 # align with that used in iDEM
                                               )
                with open(opt.proj_path + '/tvd.txt', 'a') as f:
                    f.write(str(tvd) + '\n')
if __name__ == '__main__':
    main()

