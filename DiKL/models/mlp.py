import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from models.dm_utils import make_beta_schedule


class TimeEmbedding(nn.Module):

    def __init__(self, t_embed_dim, scale=30.0):
        super().__init__()

        self.register_buffer("w", torch.randn(t_embed_dim//2)*scale)

    def forward(self, t):
        # t: (B, )
        t_proj = 2.0 * torch.pi * self.w[None, :] * t[:, None]  # (B, E//2)
        t_embed = torch.cat([torch.sin(t_proj), torch.cos(t_proj)], dim=-1)  # (B, E)
        return t_embed

class ConditionalLinear(nn.Module):
    def __init__(self, num_in, num_out):
        super(ConditionalLinear, self).__init__()
        self.num_out = num_out
        self.lin = nn.Linear(num_in, num_out)
        self.embed = TimeEmbedding(num_out)

    def forward(self, x, y):
        out = self.lin(x)
        if y is not None:
            y=y.to(x.device)
            gamma = self.embed(y)
            out = gamma.view(-1, self.num_out) + out
        return out
        
class ConditionalModel(nn.Module):
    def __init__(self, x_dim, layer_num=5, h_dim=400,learn_std=False):
        super(ConditionalModel, self).__init__()

        input_dim = x_dim

        self.lins = nn.ModuleList()
        for i in range(layer_num-1):
            if i == 0:
                self.lins.append(
                    ConditionalLinear(input_dim, h_dim)
                )
            else:
                self.lins.append(
                    ConditionalLinear(h_dim, h_dim)
                )

        self.out_mu = nn.Linear(h_dim, x_dim)
        self.learn_std=learn_std
        if learn_std:
            self.out_std = nn.Linear(h_dim, x_dim)

    def forward(self, x, y):
        for lin in self.lins: 
            x = F.silu(lin(x, y))
        x_mu = self.out_mu(x)
        if self.learn_std:
            x_std = torch.nn.functional.softplus(self.out_std(x))
            return x_mu, x_std
        return x_mu

class DiffusionModel(nn.Module):
    def __init__(self, 
                 num_steps=200, 
                 x_dim=2, 
                 layer_num=3, 
                 h_dim=400, 
                 schedule='sigmoid', 
                 device="cpu", 
                 start=1e-5, 
                 end=1e-2, 
                 ):
        super(DiffusionModel, self).__init__()
        self.num_steps = num_steps
        self.device=device
        self.betas = make_beta_schedule(schedule=schedule, n_timesteps=num_steps, start=start, end=end).to(device)

        self.alphas = 1 - self.betas
        alphas_prod = torch.cumprod(self.alphas, 0)
        self.alphas_bar_sqrt = torch.sqrt(alphas_prod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)
        self.score = ConditionalModel(x_dim, layer_num, h_dim)

    def forward(self, t, x):
        return self.score(x, t)

    
class network(nn.Module):
    def __init__(self,  input_dim=2, h_dim=400, output_dim=2, layer_num=4, learn_std=False):
        super().__init__()
        self.h_dim=h_dim
        self.input_dim=input_dim
        self.learn_std=learn_std
        self.acti=F.silu
        
        self.layers=nn.ModuleList()
        for i in range(layer_num-1):
            if i==0:
                self.layers.append(nn.Linear(input_dim, h_dim))
            else:
                self.layers.append(nn.Linear(h_dim, h_dim))

        if self.learn_std:
            self.out1 = nn.Linear(self.h_dim, output_dim)
            self.out2 = nn.Linear(self.h_dim, output_dim)
        else:
            self.out = nn.Linear(self.h_dim, output_dim)

    def forward(self,x):
        x=x.view(-1,self.input_dim)
        h=x
        for layer in self.layers:
            h=self.acti(layer(h))
        if self.learn_std:
            mu=self.out1(h)
            std=torch.nn.functional.softplus(self.out2(h))
            return mu, std
        else:
            return self.out(h)
            
class LVM(nn.Module):
    def __init__(self, z_dim,h_dim, x_dim, layer_num, device="cpu"):
        super().__init__()
        self.decoder = network(z_dim, h_dim, x_dim, layer_num, learn_std=False).to(device)

    def forward(self, z):
        x = self.decoder(z)
        return x


