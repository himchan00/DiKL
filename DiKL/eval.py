import torch
import numpy as np

import os
import yaml
import argparse
from types import SimpleNamespace as config

from train_utils import get_network, get_target, get_sample, total_variation_distance
from models.dm_utils import remove_mean
from evaluation.metric import compute_distribution_distances, total_variation_distance, get_distance

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Configuration for model training")

    parser.add_argument("--target", type=str, default='mog')
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--sample_path", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)

    parser.add_argument("--baseline_sample_dir", type=str, default=None, help="directory to save baseline samples. ")
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
    opt.model_path = args.model_path
    opt.sample_path = args.sample_path
    opt.save_dir = args.save_dir

    target = get_target(opt)

    if opt.model_path is not None:
        # if the model is provided, load model
        lvm, _ = get_network(opt)
        lvm.load_state_dict(torch.load(opt.model_path))
        # generate samples by the model
        lvm.eval()
        dikl_samples = get_sample(lvm, opt, True, opt.eval_samples)
    
    if opt.sample_path is not None:
        process = (lambda x: remove_mean(x, n_particles=opt.n_particles, n_dimensions=opt.n_dim)) if opt.e3 else (lambda x: x)
        dikl_samples = torch.load(opt.sample_path).to(opt.device)
        dikl_samples = process(dikl_samples)

    # load reference samples
    val_sample = np.load(opt.val_data_path)
    val_sample = process(torch.from_numpy(val_sample).to(opt.device))

    # load baseline samples
    try:
        fab_sample_files = [f for f in os.listdir(opt.baseline_sample_dir) if 'fab' in f]
        fab_sample = process(torch.load(os.path.join(opt.baseline_sample_dir, fab_sample_files[0])).to(opt.device))
    except:
        print('FAB sample not found')
        fab_sample = None

    try:
        idem_sample_files = [f for f in os.listdir(opt.baseline_sample_dir) if 'idem' in f]
        idem_sample = process(torch.load(os.path.join(opt.baseline_sample_dir, idem_sample_files[0])).to(opt.device))
    except:
        print('iDEM sample not found')
        idem_sample = None

    try:
        kl_sample_files = [f for f in os.listdir(opt.baseline_sample_dir) if 'kl' in f and 'dikl' not in f]
        kl_sample = process(torch.load(os.path.join(opt.baseline_sample_dir, kl_sample_files[0])).to(opt.device))
    except:
        print('KL sample not found')
        kl_sample = None

    # evaluate metrics
    def get_w2_with_bootstrapped_var(samples):
        W2 = []
        for _ in range(10):
            w2 = compute_distribution_distances(opt, samples[np.random.choice(samples.shape[0], 2000, False)][:, None], val_sample[np.random.choice(val_sample.shape[0], 2000, False)][:, None], is_molecule=opt.e3)[1][1]
            W2.append(w2)
        return np.mean(W2), np.std(W2)
    
    def get_energy_tvd_with_bootstrapped_var(samples):
        TVD = []
        for _ in range(10):
            s1 = target.energy(samples[np.random.choice(samples.shape[0], 2000, False)])
            s2 = target.energy(val_sample[np.random.choice(val_sample.shape[0], 2000, False)])
            tvd = total_variation_distance(s1, s2, bins=200)
            TVD.append(tvd)
        return np.mean(TVD), np.std(TVD)

    def get_dist_tvd_with_bootstrapped_var(samples):
        TVD = []
        for _ in range(10):
            s1 = get_distance(samples[np.random.choice(samples.shape[0], 2000, False)], opt)
            s2 = get_distance(val_sample[np.random.choice(val_sample.shape[0], 2000, False)], opt)
            tvd = total_variation_distance(s1, s2, bins=200)
            TVD.append(tvd)
        return np.mean(TVD), np.std(TVD)
    
    with open(opt.save_dir + args.target + '.txt', 'w') as f:
        w2, w2_std = get_w2_with_bootstrapped_var(dikl_samples)
        energy_tvd, energy_tvd_std = get_energy_tvd_with_bootstrapped_var(dikl_samples)
        f.write('DiKL (ours):\n')
        f.write(f'W2: {w2} {w2_std}\n')
        f.write(f'Energy TVD: {energy_tvd} {energy_tvd_std}\n')
        if opt.e3:
            dist_tvd, dist_tvd_std = get_dist_tvd_with_bootstrapped_var(dikl_samples)
            f.write(f'Distance TVD: {dist_tvd} {dist_tvd_std}\n')

        if fab_sample is not None:
            w2, w2_std = get_w2_with_bootstrapped_var(fab_sample)
            energy_tvd, energy_tvd_std = get_energy_tvd_with_bootstrapped_var(fab_sample)
            f.write('FAB:\n')
            f.write(f'W2: {w2} {w2_std}\n')
            f.write(f'Energy TVD: {energy_tvd} {energy_tvd_std}\n')
            if opt.e3:
                dist_tvd, dist_tvd_std = get_dist_tvd_with_bootstrapped_var(dikl_samples)
                f.write(f'Distance TVD: {dist_tvd} {dist_tvd_std}\n')

        if idem_sample is not None:
            w2, w2_std = get_w2_with_bootstrapped_var(idem_sample)
            energy_tvd, energy_tvd_std = get_energy_tvd_with_bootstrapped_var(idem_sample)
            f.write('iDEM:\n')
            f.write(f'W2: {w2} {w2_std}\n')
            f.write(f'Energy TVD: {energy_tvd} {energy_tvd_std}\n')
            if opt.e3:
                dist_tvd, dist_tvd_std = get_dist_tvd_with_bootstrapped_var(dikl_samples)
                f.write(f'Distance TVD: {dist_tvd} {dist_tvd_std}\n')
        
        if kl_sample is not None:
            w2, w2_std = get_w2_with_bootstrapped_var(kl_sample)
            energy_tvd, energy_tvd_std = get_energy_tvd_with_bootstrapped_var(kl_sample)
            f.write('KL:\n')
            f.write(f'W2: {w2} {w2_std}\n')
            f.write(f'Energy TVD: {energy_tvd} {energy_tvd_std}\n')
            if opt.e3:
                dist_tvd, dist_tvd_std = get_dist_tvd_with_bootstrapped_var(dikl_samples)
                f.write(f'Distance TVD: {dist_tvd} {dist_tvd_std}\n')



    # to have a nicer plot, we thin the samples
    if target.name == 'lj':
        sample_size = 1000 
    if target.name == 'dw':
        sample_size = 5000
    if target.name == 'mw':
        sample_size = 1500
    val_sample = val_sample[np.random.choice(val_sample.shape[0], sample_size, replace=False)]
    dikl_samples = dikl_samples[np.random.choice(dikl_samples.shape[0], sample_size, replace=False)]
    if fab_sample is not None:
        fab_sample = fab_sample[np.random.choice(fab_sample.shape[0], sample_size, replace=False)]
    if idem_sample is not None:
        idem_sample = idem_sample[np.random.choice(idem_sample.shape[0], sample_size, replace=False)]
    if kl_sample is not None:
        kl_sample = kl_sample[np.random.choice(kl_sample.shape[0], sample_size, replace=False)]
        

    # plot 
    plt.rcParams['figure.figsize'] = [5., 1.4]
    plt.subplot(1, 2, 1)
    if args.target == 'lj':
        min_energy = -60
        max_energy = 0
    if args.target == 'dw':
        min_energy = -26
        max_energy = -10
    if args.target == 'mw':
        min_energy = 10
        max_energy = 70
    with torch.no_grad():                   
        gt_energy = target.energy(val_sample.to(device=opt.device)).detach().cpu().numpy()
        plt.hist(gt_energy, bins=100,
                density=True,
                alpha=0.7,
                range=(min_energy, max_energy),
                color="tab:green",
                histtype="step",
                linewidth=1.4,
                label="test data",)
        model_energy = target.energy(dikl_samples).detach().cpu().numpy()
        plt.hist(model_energy[model_energy <= 300], bins=100,
                density=True,
                alpha=0.7,
                range=(min_energy, max_energy),
                color="tab:blue",
                histtype="step",
                linewidth=1.4,
                label="DiKL (ours)",)
        if fab_sample is not None:
            fab_energy = target.energy(fab_sample).detach().cpu().numpy()
            plt.hist(fab_energy, bins=100,
                    density=True,
                    alpha=0.7,
                    range=(min_energy, max_energy),
                    color="tab:red",
                    histtype="step",
                    linewidth=1.4,
                    label="FAB",)
        if idem_sample is not None:
            idem_energy = target.energy(idem_sample).detach().cpu().numpy()
            plt.hist(idem_energy, bins=100,
                    density=True,
                    alpha=0.7,
                    range=(min_energy, max_energy),
                    color="tab:orange",
                    histtype="step",
                    linewidth=1.4,
                    label="iDEM",)
        if kl_sample is not None:
            kl_energy = target.energy(kl_sample).detach().cpu().numpy()
            plt.hist(kl_energy, bins=100,
                    density=True,
                    alpha=0.7,
                    range=(min_energy, max_energy),
                    color="gray",
                    histtype="step",
                    linewidth=1.4,
                    label="KL",)
    plt.xlabel('Energy')
    plt.ylabel('Density')
    plt.legend(bbox_to_anchor=(2.2, 1.3), ncol=4, fontsize=8.5)
    if args.target in ['lj', 'dw']:
        plt.subplot(1, 2, 2)
        if args.target == 'lj':
            min_dist = 0.5
            max_dist = 4.5
        if args.target == 'dw':
            min_dist = 1.0
            max_dist = 6.5
        with torch.no_grad():
            val_dist = get_distance(val_sample, opt)
            plt.hist(val_dist, 
                    bins=100,
                    alpha=0.7,
                    density=True,
                    histtype="step",
                    linewidth=1.4,
                    color="tab:green",
                    range=(min_dist, max_dist),
                    label="test data",)
            dikl_dist = get_distance(dikl_samples, opt)
            plt.hist(dikl_dist, 
                    bins=100,
                    alpha=0.7,
                    density=True,
                    histtype="step",
                    linewidth=1.4,
                    color="tab:blue",
                    range=(min_dist, max_dist),
                    label="DiKL (ours)",)
            if fab_sample is not None:
                fab_dist = get_distance(fab_sample, opt)
                plt.hist(fab_dist, 
                        bins=100,
                        alpha=0.7,
                        density=True,
                        histtype="step",
                        linewidth=1.4,
                        color="tab:red",
                        range=(min_dist, max_dist),
                        label="FAB",)
            if idem_sample is not None:
                idem_dist = get_distance(idem_sample, opt)
                plt.hist(idem_dist, 
                        bins=100,
                        alpha=0.7,
                        density=True,
                        histtype="step",
                        linewidth=1.4,
                        color="tab:orange",
                        range=(min_dist, max_dist),
                        label="iDEM",)
            if kl_sample is not None:
                kl_dist = get_distance(kl_sample, opt)
                plt.hist(kl_dist, 
                        bins=100,
                        alpha=0.7,
                        density=True,
                        histtype="step",
                        linewidth=1.4,
                        color="gray",
                        range=(min_dist, max_dist),
                        label="KL",)
        plt.xlabel("Interatomic Distance")
    plt.savefig(args.save_dir + args.target + '.pdf', bbox_inches = 'tight')
    plt.close()



if __name__ == '__main__':
    main()

