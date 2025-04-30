import torch
import numpy as np

import os
import yaml
import argparse
from types import SimpleNamespace as config

from train_utils import get_network, get_sample
from models.dm_utils import remove_mean
import random


def parse_args():
    parser = argparse.ArgumentParser(description="Configuration for Sample Generation")

    parser.add_argument("--target", type=str, default='mog')
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--n_samples", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=1000)
    args = parser.parse_args()

    return args

def main():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    args = parse_args()

    if args.target == 'mog':
        cfg = 'configs/mog.yaml'
    elif args.target == 'mw':
        cfg = 'configs/mw.yaml'
    elif args.target == 'dw':
        cfg = 'configs/dw.yaml'
    elif args.target == 'lj':
        cfg = 'configs/lj.yaml'
    elif args.target == 'lj55':
        cfg = 'configs/lj55.yaml'
    else:
        raise NotImplementedError
    with open(cfg, 'r') as file:
        opt = config(**yaml.safe_load(file))
    opt.ais = config(**opt.ais)

    print("target:", args.target)
    
    # make dir to save results
    opt.device = args.device
    opt.model_path = args.model_path
    opt.save_dir = args.save_dir


    lvm, _ = get_network(opt)
    lvm.load_state_dict(torch.load(opt.model_path))
    # generate samples by the model
    lvm.eval()
    samples = []
    for i in range(args.n_samples // args.batch_size):
            dikl_samples = get_sample(lvm, opt, True, args.batch_size)
            samples.append(dikl_samples.cpu().numpy())
    samples = np.concatenate(samples, axis=0)
    samples = remove_mean(samples, opt.n_particles, opt.n_dim)
    # save samples
    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)
    np.save(os.path.join(opt.save_dir, f'{args.target}_samples.npy'), samples)
    



if __name__ == '__main__':
    main()

