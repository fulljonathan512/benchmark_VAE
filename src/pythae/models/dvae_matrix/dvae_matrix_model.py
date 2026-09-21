import os
from typing import Optional

import deterministic_gaussian_sampling
import numpy as np
import torch
import torch.nn.functional as F

from ...data.datasets import BaseDataset
from ..base import BaseAE
from ..base.base_utils import ModelOutput
from ..nn import BaseDecoder, BaseEncoder
from ..nn.default_architectures import Encoder_DVAE_MATRIX_MLP
from .dvae_matrix_config import DVAE_MatrixConfig


class DVAE_Matrix(BaseAE):
    """Vanilla Variational Autoencoder model.

    Args:
        model_config (VAEConfig): The Variational Autoencoder configuration setting the main
        parameters of the model.

        encoder (BaseEncoder): An instance of BaseEncoder (inheriting from `torch.nn.Module` which
            plays the role of encoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

        decoder (BaseDecoder): An instance of BaseDecoder (inheriting from `torch.nn.Module` which
            plays the role of decoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

    .. note::
        For high dimensional data we advice you to provide you own network architectures. With the
        provided MLP you may end up with a ``MemoryError``.
    """

    def __init__(
        self,
        model_config: DVAE_MatrixConfig,
        encoder: Optional[BaseEncoder] = None,
        decoder: Optional[BaseDecoder] = None,
    ):
        BaseAE.__init__(self, model_config=model_config, decoder=decoder)

        self.model_name = "VAE"

        if encoder is None:
            if model_config.input_dim is None:
                raise AttributeError(
                    "No input dimension provided !"
                    "'input_dim' parameter of BaseAEConfig instance must be set to 'data_shape' "
                    "where the shape of the data is (C, H, W ..). Unable to build encoder "
                    "automatically"
                )

            encoder = Encoder_DVAE_MATRIX_MLP(model_config)
            self.model_config.uses_default_encoder = True

        else:
            self.model_config.uses_default_encoder = False

        self.set_encoder(encoder)

    def forward(self, inputs: BaseDataset, **kwargs):
        """
        The VAE model

        Args:
            inputs (BaseDataset): The training dataset with labels

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters

        """

        x = inputs["data"]

        encoder_output = self.encoder(x)

        # log_tri_cov L: Covariance-matrix C = exp(L)*exp(L^T) in flatten version
        mu, log_tri_cov = encoder_output.embedding, encoder_output.tri_cov
        
        dirac_mixture_points = 2
        z, cov, tri_cov = self._sample_deterministic_gauss(mu, log_tri_cov, dirac_mixture_points) # third parameter with number of dirac-mixture points
        recon_x = self.decoder(z)["reconstruction"] # higher dimensional dependent on number of dirac-mixture points

        
        loss, recon_loss, kld = self.loss_function(recon_x, x, mu, cov, tri_cov , z, dirac_mixture_points)

        output = ModelOutput(
            recon_loss=recon_loss,
            reg_loss=kld,
            loss=loss,
            recon_x=recon_x,
            z=z,
        )

        return output

    def loss_function(self, recon_x, x, mu, cov, tri_cov, z, dirac_mixture_points):
        x = x.repeat_interleave(dirac_mixture_points,dim=0)
        if self.model_config.reconstruction_loss == "mse":
            recon_loss = 0.5 * F.mse_loss(
                recon_x.reshape(x.shape[0], -1),
                x.reshape(x.shape[0], -1),
                reduction="none",
            ).sum(dim=-1)

        elif self.model_config.reconstruction_loss == "bce":
            recon_loss = F.binary_cross_entropy(
                recon_x.reshape(x.shape[0], -1),
                x.reshape(x.shape[0], -1),
                reduction="none",
            ).sum(dim=-1)
        KLD = (-0.5 * torch.sum(-torch.diagonal(cov, dim1=1, dim2=2) + 1 - mu.pow(2) + torch.log(torch.square(torch.diagonal(tri_cov, dim1=1, dim2=2))), dim=-1)).repeat_interleave(dirac_mixture_points, dim=0)
        
        return (recon_loss + KLD).mean(dim=0), recon_loss.mean(dim=0), KLD.mean(dim=0)

    def _sample_deterministic_gauss(self, mu, log_tri_cov, numberOfDiracMixture): # third parameter with number of dirac-mixture points
        a = torch.tensor(())
        covs = []
        tri_covs = []
        for t in log_tri_cov:
          dim = mu.size(dim=1)
          tri_cov = torch.zeros((dim,dim))
          size = tri_cov.size(0)
          indices = torch.tril_indices(row=size, col=size)
          indices = indices.to(tri_cov.device)
          t = torch.exp(t)
          t = t.to(tri_cov.device)
          tri_cov[indices[0],indices[1]] = t
          cov = torch.mm(tri_cov, torch.transpose(tri_cov,0,1))
          numpyCovr = cov.cpu().detach().numpy()
          approx = np.zeros((numberOfDiracMixture, mu.size(dim=1)))
          g2d = deterministic_gaussian_sampling.GaussianToDiracApproximation()
          g2d.approximate_double(numpyCovr, numberOfDiracMixture, mu.size(dim = 1), approx)
          del g2d
          a = torch.cat((a,torch.from_numpy(approx)),0)
          tri_covs.append(tri_cov)
          covs.append(cov)

        mu = torch.repeat_interleave(mu,repeats=numberOfDiracMixture,dim=0)
        b = a.cuda()
        covs = torch.stack(covs)
        covs = covs.to(mu.device)
        tri_covs = torch.stack(tri_covs)
        tri_covs = tri_covs.to(mu.device)
        
        return (mu + b).float(), covs, tri_covs

    def get_nll(self, data, n_samples=1, batch_size=100):
        """
        Function computed the estimate negative log-likelihood of the model. It uses importance
        sampling method with the approximate posterior distribution. This may take a while.

        Args:
            data (torch.Tensor): The input data from which the log-likelihood should be estimated.
                Data must be of shape [Batch x n_channels x ...]
            n_samples (int): The number of importance samples to use for estimation
            batch_size (int): The batchsize to use to avoid memory issues
        """

        if n_samples <= batch_size:
            n_full_batch = 1
        else:
            n_full_batch = n_samples // batch_size
            n_samples = batch_size

        log_p = []

        for i in range(len(data)):
            x = data[i].unsqueeze(0)

            log_p_x = []

            for j in range(n_full_batch):
                x_rep = torch.cat(batch_size * [x])

                encoder_output = self.encoder(x_rep)
                mu, log_var = encoder_output.embedding, encoder_output.log_covariance

                std = torch.exp(0.5 * log_var)
                z, _ = self._sample_deterministic_gauss(mu, std,1)

                log_q_z_given_x = -0.5 * (
                    log_var + (z - mu) ** 2 / torch.exp(log_var)
                ).sum(dim=-1)
                log_p_z = -0.5 * (z**2).sum(dim=-1)

                recon_x = self.decoder(z)["reconstruction"]

                if self.model_config.reconstruction_loss == "mse":
                    log_p_x_given_z = -0.5 * F.mse_loss(
                        recon_x.reshape(x_rep.shape[0], -1),
                        x_rep.reshape(x_rep.shape[0], -1),
                        reduction="none",
                    ).sum(dim=-1) - torch.tensor(
                        [np.prod(self.input_dim) / 2 * np.log(np.pi * 2)]
                    ).to(
                        data.device
                    )  # decoding distribution is assumed unit variance  N(mu, I)

                elif self.model_config.reconstruction_loss == "bce":
                    log_p_x_given_z = -F.binary_cross_entropy(
                        recon_x.reshape(x_rep.shape[0], -1),
                        x_rep.reshape(x_rep.shape[0], -1),
                        reduction="none",
                    ).sum(dim=-1)

                log_p_x.append(
                    log_p_x_given_z + log_p_z - log_q_z_given_x
                )  # log(2*pi) simplifies

            log_p_x = torch.cat(log_p_x)

            log_p.append((torch.logsumexp(log_p_x, 0) - np.log(len(log_p_x))).item())
        return np.mean(log_p)
