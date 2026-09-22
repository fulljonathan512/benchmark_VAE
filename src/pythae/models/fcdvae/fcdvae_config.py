from pydantic.dataclasses import dataclass
from typing_extensions import Literal

from ..base.base_config import BaseAEConfig


@dataclass
class FCDVAEConfig(BaseAEConfig):
    """FCDVAE config class.

    Parameters:
        input_dim (tuple): The input_data dimension.
        latent_dim (int): The latent space dimension. Default: None.
        reconstruction_loss (str): The reconstruction loss to use ['bce', 'mse']. Default: 'mse'
        gaussian_approximation (str): The kind of approximation calcutlation for Gaussian density points ['lcd', 'fib']. Default: 'lcd'
        number_gaussian_points (int): Amount of approximation points calculated. Default: 0
    """

    reconstruction_loss: Literal["bce", "mse"] = "mse"
    gaussian_approximation: Literal["lcd", "fib"] = "lcd"
    number_gaussian_points: int = 1
