import torch
import torch.optim as optim
from torch.distributions import Normal, Gumbel
import numpy as np

import os
import yaml
import argparse
from tqdm import tqdm
from types import SimpleNamespace as config

from sampler.denoise_sampler import diffusion_sampler
from sampler.ais import DenoisingLangevinDynamics



def parse_args():
    parser = argparse.ArgumentParser(description="Configuration for model training")

    parser.add_argument("--target", type=str, default='LJ')
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    if args.target == 'mog':
        opt = yaml.safe_load('configs/mog.yaml')
    elif args.target == 'mw':
        opt = yaml.safe_load('configs/mw.yaml')
    elif args.target == 'dw':
        opt = yaml.safe_load('configs/dw.yaml')
    elif args.target == 'lj':
        opt = yaml.safe_load('configs/lj.yaml')
    else:
        raise NotImplementedError
    
    # make dir to save results
    opt.device = args.device
    opt.proj_path = opt.save_path + opt.name

    if os.path.exists(opt.proj_path):
        counter = 1
        # Loop until a non-existing path is found
        while os.path.exists(opt.proj_path):
            opt.proj_path = opt.save_path + opt.name + f"_{counter}"
            counter += 1

    os.makedirs(opt.proj_path, exist_ok=True)
    os.makedirs(opt.proj_path+'/model', exist_ok=True)
    os.makedirs(opt.proj_path+'/plot', exist_ok=True)



    


    score_model = EGNN_dynamics(n_particles=opt.n_particles,
                                n_dimension=opt.n_dim,
                                hidden_nf=144,
                                act_fn=torch.nn.ReLU(),
                                n_layers=8,
                                recurrent=True,
                                attention=True,
                                condition_time=True,
                                tanh=True,
                                agg="sum",
                                energy_function=None,
                                device=opt.device,
                                schedule=opt.dsm_scheme,
                                num_steps=opt.Tmax,
                                start=opt.start, 
                                end=opt.end).to(opt.device) # noise prediction

    lvm = EGNN_dynamics(n_particles=opt.n_particles,
                        n_dimension=opt.n_dim,
                        hidden_nf=144,
                        act_fn=torch.nn.ReLU(),
                        n_layers=8,
                        recurrent=True,
                        attention=True,
                        condition_time=False,
                        tanh=True,
                        agg="sum",
                        energy_function=None).to(opt.device)
    lvm_optim = optim.Adam(lvm.parameters(), lr=opt.lvm_lr)
    score_optim = optim.Adam(score_model.parameters(), lr=opt.score_lr)


    def dsm_loss(model, x, opt):
        batch_size = x.shape[0]
        t=torch.randint(1, model.num_steps,[batch_size], device=x.device)

        a = extract(model.alphas_bar_sqrt, t, x)
        sigma = extract(model.one_minus_alphas_bar_sqrt, t, x)
        e = remove_mean(torch.randn_like(x), n_particles=opt.n_particles, n_dimensions=3)
        x0 = x
        x = x * a + e * sigma
        
        if opt.network_param == 'noise':
            return (e - model(t, x)).square().sum(-1).mean(0)
        elif opt.network_param == 'score':
            score = model(t, x)
            return (e + sigma * score).square().sum(-1).mean(0)
        elif opt.network_param == 'mean':
            score = (x - a * model(t, x)) / sigma**2
            return (e + sigma * score).square().sum(-1).mean(0)
                
    best_model_energy = 100

    for it in tqdm(range(1,opt.max_iter+1)):

        # train LVM score network

        lvm.eval()
        score_model.train()
        Score_loss = []
        for _ in range(opt.score_iter):
            
            score_optim.zero_grad()
            z = remove_mean(torch.randn([opt.batch_size, opt.n_particles*3], device=opt.device), n_particles=opt.n_particles, n_dimensions=3)
            x = remove_mean(lvm(None, z).detach(), n_particles=opt.n_particles, n_dimensions=3)

            # dsm loss
            score_loss = dsm_loss(score_model, x, opt)# + slice_sm_loss(score_model, x, opt)

            score_loss.backward()
            score_optim.step()

            Score_loss.append(score_loss.item())
        
        # train LVM with diffusive rKL
        lvm.train()
        score_model.eval()
        lvm_optim.zero_grad()
        z = remove_mean(torch.randn([opt.batch_size, opt.n_particles*3], device=opt.device), n_particles=opt.n_particles, n_dimensions=3)
        x = remove_mean(lvm(None, z), n_particles=opt.n_particles, n_dimensions=3)

        t = torch.randint(1, score_model.num_steps,[x.shape[0]], device=x.device)
        a = extract(score_model.alphas_bar_sqrt.to(opt.device), t, x)
        sigma = extract(score_model.one_minus_alphas_bar_sqrt.to(opt.device), t, x)
        e = remove_mean(torch.randn_like(x), n_particles=opt.n_particles, n_dimensions=3)
        x_t = (x * a + e * sigma)

        x_t_clone = x_t.detach().clone()

        if opt.resample:
            x_0_init = diffusion_sampler(target_log_p = lambda x: -target.energy(x) + Normal(x*a, sigma).log_prob(x_t_clone).sum(-1),
                                        ais_steps=opt.AIS_step,
                                        ais_step_size=opt.hmc_step_size,
                                        x_t=x_t_clone,
                                        bar_alpha_t=a**2,
                                        n_is=opt.n_is,
                                        hmc_step=opt.hmc_step,
                                        trunc_lower_bound=-1.0*opt.boundary,
                                        trunc_upper_bound=opt.boundary, 
                                        device=opt.device,
                                        resample=True,
                                        verbose=opt.verbose,
                                        smc_gap=opt.smc_gap,
                                        LG=True,
                                        lg_step=opt.lg_step,
                                        mean_center=lambda x: remove_mean(x, opt.n_particles, 3)
                                        )
            if opt.num_langevin_steps > 0:
                langevin_dynamics = denoising_LD_generator(lambda x: target.energy(x, smooth_=True), x_0_init.detach().clone(), x_t.detach().clone(), a, sigma, step_size=opt.langevin_step_size, mh=True, device=opt.device)
                Acc = []

                X0s = []
                GRADs = []

                for lg_it in range(opt.num_langevin_steps):
                    x_0, acc, gradx0 = langevin_dynamics.sample()
                    Acc.append(acc)

                    if lg_it >= 500:
                        x_0 = remove_mean(x_0, n_particles=opt.n_particles, n_dimensions=3).detach().clone()
                        gradx0 = remove_mean(gradx0, n_particles=opt.n_particles, n_dimensions=3).detach().clone()
                        X0s.append(x_0)
                        GRADs.append(gradx0)

                if np.mean(Acc) > 0.6:
                    opt.langevin_step_size *= 1.5
                elif np.mean(Acc) < 0.5:
                    opt.langevin_step_size /= 1.5
            else:
                x_0 = x_0_init
            if opt.verbose:
                print('LG ACC %.3f'%acc)

            x_0 = torch.stack(X0s, 0).mean(0)
            _target_score = -torch.stack(GRADs, 0).mean(0)

            if opt.grad_estimator == 'msm':
                target_score = a * (x_0 + _target_score) - x_t
            elif opt.grad_estimator == 'tsm':
                target_score = _target_score / a
            else:
                raise NotImplementedError
            # print(target_score.shape)
        else:
            raise NotImplementedError
            x_0, is_weight = diffusion_sampler(target_log_p = lambda x: -target.energy(x) + Normal(x*a, sigma).log_prob(x_t_clone).sum(-1),
                                        ais_steps=opt.AIS_step,
                                        ais_step_size=opt.hmc_step_size,
                                        x_t=x_t_clone,
                                        bar_alpha_t=a**2,
                                        n_is=opt.n_is,
                                        hmc_step=opt.hmc_step,
                                        trunc_lower_bound=-1.0*opt.boundary,
                                        trunc_upper_bound=opt.boundary, 
                                        device=opt.device,
                                        resample=0,
                                        verbose=opt.verbose,
                                        smc_gap=opt.smc_gap,
                                        LG=True,
                                        lg_step=opt.lg_step,
                                        mean_center=lambda x: remove_mean(x, opt.n_particles, 3)
                                        )
        
            x_0_clone = x_0.detach().clone().requires_grad_(True)
            (target.log_prob(x_0_clone, smooth_=False).sum()).backward()
            _target_score = x_0_clone.grad.detach().clone()
            if opt.grad_estimator == 'msm':
                target_score = a * (x_0 + _target_score) - x_t
            elif opt.grad_estimator == 'tsm':
                target_score = _target_score / a
            else:
                raise NotImplementedError
            target_score = (target_score * torch.nn.Softmax(0)(is_weight)[:, :, None]).sum(dim=0)
            # print(target_score.shape)
            
        if opt.network_param == 'noise':
            lvm_noise = score_model(t, x_t)
            lvm_score = -lvm_noise / sigma
        elif opt.network_param == 'score':
            lvm_score = score_model(t, x_t)
        elif opt.network_param == 'mean':
            lvm_score = (x - a * score_model(t, x)) / sigma**2
            
        if opt.weight=="uniform":
            w=1
        elif opt.weight=="1/a":
            w=1/a
        elif opt.weight=="linear_anneal":
            w=(it/opt.max_iter)+(1-it/opt.max_iter)*(1/a)
        elif opt.weight=="poly_anneal":
            eta = (it / opt.max_iter) ** opt.poly_coeff 
            w = eta * 1 + (1 - eta) * (1 / a)
        elif opt.weight=="s2/a":
            w=sigma**2/a
        elif opt.weight=="s2/a2":
            w=sigma**2/a**2
        elif opt.weight=="a":
            w=a
        else:
            raise 

            
        score_diff = w*(lvm_score - target_score).detach()
        lvm_loss=(score_diff*x_t).sum(1).mean()

        if ~(torch.isnan(lvm_loss) | torch.isinf(lvm_loss)):
            lvm_loss.backward()

        torch.nn.utils.clip_grad_norm_(lvm.parameters(), opt.grad_norm_clip)
        lvm_optim.step()


        if it==1 or it%opt.show_iter==0:
            print('Checking...')
            plt.rcParams['figure.figsize'] = [16, 4]
            plt.subplot(1, 3, 1)
            lvm.eval()
            with torch.no_grad():
                gt_energy = target.energy(torch.from_numpy(val_data).to(device=opt.device), break_symmetry=False, smooth_=False).detach().cpu().numpy()
                plt.hist(gt_energy, np.linspace(-70, 20, 100), density=True, alpha=1, histtype='step')

                z = torch.randn([2000, opt.n_particles*opt.n_dim], device=opt.device)
                x = lvm(None, z)

                model_energy = target.energy(x, break_symmetry=False, smooth_=False).detach().cpu().numpy()
                plt.hist(model_energy[model_energy <= 300], np.linspace(-70, 20, 100), density=True, alpha=1, histtype='step')

                if opt.resample:
                    x_0_resample = X0s[-1].detach().cpu()
                else:
                    gumbeled_density_ratio = is_weight + Gumbel(torch.tensor(0.0, device=opt.device), torch.tensor(1.0, device=opt.device)).sample(is_weight.shape)
                    idx = gumbeled_density_ratio.argmax(0)
                    x_0_resample = x_0.permute(1, 0, 2)[np.arange(x_0.shape[1]), idx, :].detach().cpu()

                posterior_energy = target.energy(x_0_resample, break_symmetry=False, smooth_=False).detach().cpu().numpy()
                plt.hist(posterior_energy, np.linspace(-70, 20, 100), density=True, alpha=1, histtype='step')
            plt.xlim(-70, 20)
            # plt.xlim(0, 120)
            plt.subplot(1, 3, 2)
            
            
            a = (((val_data.reshape(-1, opt.n_particles, 1, opt.n_dim) - val_data.reshape(-1, 1, opt.n_particles, opt.n_dim))**2).sum(-1)**0.5)
            b = (((x.reshape(-1, opt.n_particles, 1, opt.n_dim) - x.reshape(-1, 1, opt.n_particles, opt.n_dim))**2).sum(-1).sqrt()).cpu()
            c = (((x_0_resample.reshape(-1, opt.n_particles, 1, opt.n_dim) - x_0_resample.reshape(-1, 1, opt.n_particles, opt.n_dim))**2).sum(-1).sqrt()).cpu()
            diaga = torch.triu_indices(a.shape[1], a.shape[1], 1)
            diagb = torch.triu_indices(b.shape[1], b.shape[1], 1)
            diagc = torch.triu_indices(c.shape[1], c.shape[1], 1)

            plt.hist(a[:, diaga[0], diaga[1]].flatten(), 100, density=1, alpha=1, histtype='step',)
            plt.hist(b[:, diagb[0], diagb[1]].flatten(), 100, density=1, alpha=1, histtype='step',)
            plt.hist(c[:, diagc[0], diagc[1]].flatten(), 100, density=1, alpha=1, histtype='step',)
            plt.xlim(0, 5)

            plt.subplot(1, 3, 3)

            plt.plot(Score_loss)
            
            print(it, 'GT Energy %.3f'%gt_energy.mean(),
                'Model Energy %.3f'%model_energy[model_energy <= 300].mean(),
                'Posterior Energy %.3f'%posterior_energy.mean(), flush=True)

            plt.savefig(opt.proj_path+'/plot/show_plot%d.png'%it)
            plt.close()

            if model_energy[model_energy <= 300].mean() <= best_model_energy:
                best_model_energy = model_energy[model_energy <= 300].mean()
                # save ckpt
                torch.save(lvm.state_dict(), 'acc_smooth_128_1000LG_center_LJ13_LVM.pt')
                torch.save(score_model.state_dict(), 'acc_smooth_128_1000LG_center_LJ13_LVM_SCORE.pt')
                print('Save New ckpt with average energy %.3f'%model_energy[model_energy <= 300].mean(), flush=True)


def denoising_LD_generator(energy, x, tx, alpha, std, step_size, mh, device):
    return DenoisingLangevinDynamics(x, 
                                    energy,
                                    lambda z: ((tx-alpha*z)**2/(2*std**2)).sum(-1),
                                    step_size=step_size,
                                    mh=mh,
                                    device=device)
if __name__ == '__main__':
    main()
