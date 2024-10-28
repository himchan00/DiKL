from energy.mog40 import *
from energy.dw import *
from energy.lj import *
from energy.mw import *


def get_target(opt):
    if opt.name == 'mog':
        target = GMM(dim=2, n_mixes=40,loc_scaling=40, log_var_scaling=1,device=opt.device)
        target.to(opt.device)
    elif opt.name == 'dw':

    class Target(LennardJonesPotential):
        def log_prob(self, x, break_symmetry=True, smooth_=False):
            assert x.shape[-1] == opt.n_particles * opt.n_dim
            bsz = x.shape[:-1]
            x = x.reshape(-1, opt.n_particles, opt.n_dim)
            return super().log_prob(x, smooth_=smooth_).squeeze(-1).reshape(*bsz)# + x_mean_logp
        def energy(self, x, break_symmetry=True, smooth_=False):
            assert x.shape[-1] == opt.n_particles * opt.n_dim
            bsz = x.shape[:-1]
            x = x.reshape(-1, opt.n_particles, opt.n_dim)
            return -super().log_prob(x, smooth_=smooth_).squeeze(-1).reshape(*bsz)
    target = Target(opt.n_particles*opt.n_dim,
                    n_particles=opt.n_particles,
                    eps=1.0,
                    rm=1.0,
                    oscillator=True,
                    oscillator_scale=1.0,
                    two_event_dims=True,
                    energy_factor=1.0)
    
    return target