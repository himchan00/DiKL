from energy.mog40 import *
from energy.dw import *
from energy.lj import *
from energy.mw import *

from models.mlp import *
from models.egnn import *

def get_target(opt):
    """
    Return target objective.
    """
    if opt.name == 'mog':
        target = GMM(dim=2, n_mixes=40,loc_scaling=40, log_var_scaling=1,device=opt.device)
        target.to(opt.device)
    elif opt.name == 'dw':
        class Target(MultiDoubleWellEnergy):
            def log_prob(self, x):
                assert x.shape[-1] == opt.n_particles * opt.n_dim
                bsz = x.shape[:-1]
                x = x.reshape(-1, opt.n_particles * opt.n_dim)
                return super().log_prob(x).squeeze(-1).reshape(*bsz)
            def energy(self, x):
                assert x.shape[-1] == opt.n_particles * opt.n_dim
                bsz = x.shape[:-1]
                x = x.reshape(-1, opt.n_particles * opt.n_dim)
                return super().energy(x).squeeze(-1).reshape(*bsz)
        target = Target(opt.n_particles*opt.n_dim,
                        n_particles=opt.n_particles,
                        two_event_dims=False)
    elif opt.name == 'lj':
        class Target(LennardJonesPotential):
            def log_prob(self, x, smooth_=False):
                assert x.shape[-1] == opt.n_particles * opt.n_dim
                bsz = x.shape[:-1]
                x = x.reshape(-1, opt.n_particles, opt.n_dim)
                return super().log_prob(x, smooth_=smooth_).squeeze(-1).reshape(*bsz)
            def energy(self, x, smooth_=False):
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
                        two_event_dims=True, # always set to True as we wrap it with reshape
                        energy_factor=1.0)
    elif opt.name == 'mw':
        target = ManyWellEnergy(dim=opt.n_dim)
    else:
        raise NotImplementedError
        
    return target

def get_network(opt):
    """
    Return score network and neural sampler
    """
    if opt.name in ['mog', 'mw']:
        lvm = LVM(opt.x_dim, opt.h_dim, opt.x_dim, opt.layer_num, opt.device).to(opt.device)
        score_model= DiffusionModel(
                                    x_dim=opt.n_dim,
                                    num_steps=opt.Tmax, 
                                    layer_num=opt.score_layer_num,
                                    h_dim=opt.score_h_dim,
                                    schedule=opt.dsm_scheme, 
                                    start=opt.start, 
                                    end=opt.end,
                                    device=opt.device,
                                    ).to(opt.device)
    if opt.name in ['dw', 'lj']:
        lvm = EGNN_dynamics(n_particles=opt.n_particles,
                            n_dimension=opt.n_dim,
                            hidden_nf=opt.h_dim,
                            act_fn=torch.nn.ReLU(),
                            n_layers=opt.layer_num,
                            recurrent=True,
                            attention=True,
                            condition_time=False,
                            tanh=True,
                            agg="sum",
                            energy_function=None).to(opt.device)
        score_model = EGNN_dynamics(n_particles=opt.n_particles,
                                    n_dimension=opt.n_dim,
                                    hidden_nf=opt.score_h_dim,
                                    act_fn=torch.nn.ReLU(),
                                    n_layers=opt.score_layer_num,
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
                                    end=opt.end).to(opt.device)

    return lvm, score_model

def get_sample(lvm, opt, stop_grad):
    if opt.e3:
        process = lambda x: remove_mean(x, n_particles=opt.n_particles, n_dimensions=3)
        z_dim = opt.n_particles * opt.n_dim
        sample = lambda x: lvm(None, x)
    else:
        process = lambda x: x
        z_dim = opt.n_dim
        sample = lambda x: lvm(x)
    
    z = process(torch.randn([opt.batch_size, z_dim], device=opt.device))
    if stop_grad:
        with torch.no_grad():
            x = process(sample(z).detach())
    else:
        x = process(sample(z))
        
    return x

