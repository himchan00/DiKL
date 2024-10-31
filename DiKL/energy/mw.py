import torch
import numpy as np
import matplotlib.pyplot as plt
from energy.mog40 import plot_contours

class Energy():
    def __init__(self, dim):
        super().__init__()
        self._dim = dim
        
    @property
    def dim(self):
        return self._dim
    
    def _energy(self, x):
        raise NotImplementedError()
        
    def energy(self, x, temperature=None):
        #assert x.shape[-1] == self._dim, "`x` does not match `dim`"
        if temperature is None:
            temperature = 1.
        if x.ndim == 1:
            x = x.unsqueeze(0)
        return self._energy(x) / temperature



class DoubleWellEnergy(Energy):
    def __init__(self, dim=2, a=-0.5, b=-6.0, c=1.0, k=1.0):
        super().__init__(dim)
        self._a = a
        self._b = b
        self._c = c
        self._k = k
    
    def _energy(self, x):
        d = x[..., 0:1]
        v = x[..., 1:]
        e1 = self._a * d + self._b * (d**2) + self._c * (d**4)
        e2 = torch.sum(0.5 * self._k * (v**2), dim=-1, keepdim=True)
        return e1 + e2
   
    @property
    def log_Z(self):
        if self._a == -0.5 and self._b == -6.0 and self._c == 1.0 and self._k == 1.0:
            log_Z_dim0 = np.log(11784.50927)
            log_Z_dim1 = 0.5 * np.log(2 * torch.pi)
            return log_Z_dim0 + log_Z_dim1
        else:
            raise NotImplementedError
        
    def log_prob(self, x):
        if self._a == -0.5 and self._b == -6.0 and self._c == 1.0 and self._k == 1.0:
            return -self.energy(x).squeeze(-1) - self.log_Z
        else:
            raise NotImplementedError
        

class ManyWellEnergy(Energy):
    def __init__(self, dim, a=-0.5, b=-6, c=1, k=1):
        assert dim % 2 == 0 and dim < 40
        super().__init__(dim)
        self.double_well = DoubleWellEnergy(dim=2, a=a, b=b, c=c, k=k)
        self.n_wells = dim // 2
        self.center = 1.7
        self.shallow_well_bounds = [-1.75, -1.65]
        self.deep_well_bounds = [1.7, 1.8]

    @property
    def log_Z(self):
        return torch.tensor(self.double_well.log_Z*self.n_wells)
    
    @property
    def Z(self):
        return torch.exp(self.log_Z)
    
    def _energy(self, x):
        return torch.stack(
            [self.double_well._energy(x[..., i*2:i*2+2]).squeeze(-1) for i in range(self.n_wells)], 
            dim=0
        ).sum(dim=0)
    
    def log_prob(self, x, **kwargs):
        return -self.energy(x) - self.log_Z
    

class MuellerEnergy(Energy):
    def __init__(self):
        dim = 2 
        super().__init__(dim)
        self._alpha = 0.1
        self._a = torch.FloatTensor([[-1, -1, -6.5, 0.7]])
        self._b = torch.FloatTensor([[0, 0, 11, 0.6]])
        self._c = torch.FloatTensor([[-10, -10, -6.5, 0.7]])
        self._A = torch.FloatTensor([[-200.0, -100.0, -170.0, 15.0]])
        self.d_h = torch.FloatTensor([[1.0, 0, -0.5, -1.0]])
        self.v_h = torch.FloatTensor([[0, 0.5, 1.5, 1.0]])
    
    def _energy(self, x):
        d = x[:, 0:1]
        v = x[:, 1:2]
        e = self._alpha * torch.sum(
            self._A * torch.exp(
                self._a*(d-self.d_h)**2 + self._b*(d-self.d_h)*(v-self.v_h) + self._c*(v-self.v_h)**2
            ), 
            dim=1,
            keepdim=True
        )
        return e
    

def get_target_log_prob_marginal_pair_alt(log_prob_doublewell, i: int, j: int):
    def log_prob(x):
        if i % 2 == 0:
            first_dim_x = torch.zeros_like(x)
            first_dim_x[:, 0] = x[:, 0]
        else:
            first_dim_x = torch.zeros_like(x)
            first_dim_x[:, 1] = x[:, 0]
        if j % 2 == 0:
            second_dim_x = torch.zeros_like(x)
            second_dim_x[:, 0] = x[:, 1]
        else:
            second_dim_x = torch.zeros_like(x)
            second_dim_x[:, 1] = x[:, 1]
        return log_prob_doublewell(first_dim_x) + log_prob_doublewell(second_dim_x)
    return log_prob

def plot_marginal_paris(log_prob_doublewell, samples, plotting_bounds, n_contour_levels, grid_width_n_points, save_dir, s=10, alpha=0.5):
    count = 0
    for i in range(2):
        for j in range(2):
            if count == 0:
                fig, axs = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(4*3, 3))
            
            marginal_log_prob = get_target_log_prob_marginal_pair_alt(log_prob_doublewell, i, j+2)
            plot_contours(marginal_log_prob, samples=samples, bounds=plotting_bounds, ax=axs[count], plot_marginal_dims=(i,j+2), n_contour_levels=n_contour_levels, grid_width_n_points=grid_width_n_points, s=s, alpha=alpha, plt_show=False, xy_tick=False)
            
            count += 1

    plt.savefig(save_dir)
    plt.close()