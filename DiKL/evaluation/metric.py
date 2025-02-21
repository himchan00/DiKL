import numpy as np
from typing import Union
import torch

from optimal_transport import wasserstein

"""
This function modified from iDEM's repo at https://github.com/jarridrb/DEM, released under the MIT license.
In our case, we are not using the MMD distances, so we have removed them from the code.
"""


def get_distance(sample, opt):
    _a = (((sample.reshape(-1, opt.n_particles, 1, opt.n_dim) - sample.reshape(-1, 1, opt.n_particles, opt.n_dim))**2).sum(-1)**0.5)
    _diaga = torch.triu_indices(_a.shape[1], _a.shape[1], 1)
    return _a[:, _diaga[0], _diaga[1]].flatten()

def total_variation_distance(samples1, samples_test, bins=200):
    H_data_set, x_data_set = np.histogram(samples_test, bins=bins)
    H_generated_samples, _ = np.histogram(samples1, bins=(x_data_set))
    total_var = (
        0.5
        * np.abs(
            H_data_set / H_data_set.sum() - H_generated_samples / H_generated_samples.sum()
        ).sum()
    )
    return total_var


def compute_distribution_distances(
    opt, pred: torch.Tensor, true: Union[torch.Tensor, list], is_molecule=True
):
    """computes distances between distributions.
    pred: [batch, times, dims] tensor
    true: [batch, times, dims] tensor or list[batch[i], dims] of length times

    This handles jagged times as a list of tensors.

    """
    NAMES = [
        "1-Wasserstein",
        "2-Wasserstein",
    ]
    is_jagged = isinstance(true, list)
    pred_is_jagged = isinstance(pred, list)
    dists = []
    to_return = []
    names = []
    filtered_names = [name for name in NAMES if not is_jagged or not name.endswith("MMD")]
    ts = len(pred) if pred_is_jagged else pred.shape[1]
    for t in np.arange(ts):
        if pred_is_jagged:
            a = pred[t]
        else:
            a = pred[:, t, :]
        if is_jagged:
            b = true[t]
        else:
            b = true[:, t, :]
        w1 = wasserstein(a, b, power=1)
        w2 = wasserstein(a, b, power=2)

        dists.append((w1, w2))
        # For multipoint datasets add timepoint specific distances
        if ts > 1:
            names.extend([f"t{t+1}/{name}" for name in filtered_names])
            to_return.extend(dists[-1])

    to_return.extend(np.array(dists).mean(axis=0))
    names.extend(filtered_names)
    return names, to_return

