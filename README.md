# Diffusive KL Divergence (DiKL)
Official PyTorch implementation of paper [Training Neural Samplers with Reverse Diffusive KL Divergence](https://arxiv.org/abs/2410.12456).
In this paper, we introduce Diffusive KL Divergence (DiKL), a reverse-KL-based divergence that promotes mode-covering behavior, in contrast to the standard reverse KL, which tends to focus on mode-seeking.
![](./assets/compare_crop.gif)

## Reproducing results for DiKL and Baselines
We provide samples on MoG-40, MW-32, DW-4, and LJ-13 for our methods and baselines (iDEM, FAB, reverse KL) in the following for rapid reproduction of our results.


## Environment Setup


## Training Neural Sampler with DiKL

## Evaluation

## Citation

Please cite the following paper if you use this repo:

```
@article{he2024training,
  title={Training Neural Samplers with Reverse Diffusive KL Divergence},
  author={He, Jiajun and Chen, Wenlin and Zhang, Mingtian and Barber, David and Hern{\'a}ndez-Lobato, Jos{\'e} Miguel},
  journal={arXiv preprint arXiv:2410.12456},
  year={2024}
}
```
