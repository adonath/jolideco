from functools import cached_property

import torch
from torch.distributions import Categorical, Dirichlet, MixtureSameFamily

from jolideco.utils.torch import TORCH_DEFAULT_DEVICE, uniform_torch


class DirichletMixtureModel:
    """Dirichlet mixture model.

    Attributes
    ----------
    alphas : `torch.Tensor`
        Alphas
    weights : `torch.Tensor`
        Weights
    device: `torch.Device`
        Pytorch device
    """

    def __init__(self, weights, alphas, device=TORCH_DEFAULT_DEVICE):
        log_w = torch.log(
            torch.tensor(weights.clone().detach(), requires_grad=True, device=device)
        )
        self._log_weights = torch.nn.Parameter(log_w)

        log_alpha = torch.log(
            torch.tensor(alphas.clone().detach(), requires_grad=True, device=device)
        )
        self._log_alphas = torch.nn.Parameter(log_alpha)

    @property
    def alphas(self):
        """Alphas"""
        return torch.exp(self._log_alphas)

    @property
    def weights(self):
        """Weights"""
        weights = torch.exp(self._log_weights)
        return weights / weights.sum()

    @cached_property
    def patch_shape(self):
        """Patch shape (tuple)"""
        shape_mean = self.alphas.shape
        npix = int((shape_mean[-1]) ** 0.5)
        return npix, npix

    @cached_property
    def n_features(self):
        """Number of features"""
        n_features, _ = self.alphas.shape
        return n_features

    @cached_property
    def n_components(self):
        """Number of features"""
        n_components = self.weights.shape
        return n_components

    @property
    def _dmm(self):
        mix = Categorical(self.weights)
        dirichlet = Dirichlet(self.alphas)
        dmm = MixtureSameFamily(mix, dirichlet)
        return dmm

    def estimate_log_prob(self, x):
        """Estimate log probability of x."""
        return self._dmm.log_prob(x)

    @classmethod
    def from_n_components(cls, n_components, n_features, generator=None):
        """Create a Dirichlet mixture model from n components and features.

        Parameters
        ----------
        n_components : int
            Number of components
        n_features : int
            Number of features

        Returns
        -------
        dmm : `DirichletMixtureModel`
            Dirichlet mixture model
        """
        if generator is None:
            generator = torch.Generator()

        weights = uniform_torch(
            x_min=0, x_max=1, size=(n_components,), generator=generator
        )
        weights = weights / weights.sum()
        alphas = uniform_torch(
            x_min=0, x_max=1, size=(n_components, n_features), generator=generator
        )
        return cls(weights=weights, alphas=alphas)

    def write(self, filename):
        """Write model parameters to a file."""
        pass

    @classmethod
    def read(cls, filename):
        """Read model parameters from a file."""
        pass
