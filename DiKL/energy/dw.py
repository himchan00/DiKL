import numpy as np
import torch
from bgflow import Energy
from bgflow import MultiDoubleWellPotential

class MultiDoubleWellEnergy(Energy):
    def __init__(
        self,
        dim,
        n_particles,
        two_event_dims=False,
    ):

        if two_event_dims:
            super().__init__([n_particles, dim // n_particles])
        else:
            super().__init__(dim)
        self._n_particles = n_particles
        self._n_dims = dim // n_particles

        self.multi_double_well = MultiDoubleWellPotential(
                                                        dim=dim,
                                                        n_particles=n_particles,
                                                        a=0.9,
                                                        b=-4,
                                                        c=0,
                                                        offset=4,
                                                        two_event_dims=False,
                                                    )

    def energy(self, x):
        x = x.view(-1, self._n_particles * self._n_dims)
        energies = self.multi_double_well.energy(x)

        return energies[:, None]

    def _energy_numpy(self, x):
        x = torch.Tensor(x)
        return self.energy(x).cpu().numpy()

    def log_prob(self, x):
        return -self.energy(x)


