import torch
import torch.nn as nn
import torch.nn.functional as F
import sympy
import warnings
import math
import torch.distributions as dist
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass

from .flow_components.interpolation_solver import AffineInterp
from .flow_components.train_time_sampler import TrainTimeSampler
from .flow_components.train_time_weight import TrainTimeWeight
from .flow_components.loss_function import RectifiedFlowLossFunction
from .utils import match_dim_with_data


def gm_to_sample(
        gm,
        gm_power=1,
        n_samples=1,
        cov_sharpen=False):
    """
    Args:
        gm (dict):
            means (torch.Tensor): (bs, *, num_gaussians, out_channels, h, w)
            logstds (torch.Tensor): (bs, *, 1, 1, 1, 1) or (bs, *, num_gaussians, 1, h, w)
            logweights (torch.Tensor): (bs, *, num_gaussians, 1, h, w)
            or
            means (torch.Tensor): (bs, *, num_gaussians, h, w, out_channels)
            covs (torch.Tensor): (bs, *, 1 or num_gaussians, h, w, out_channels, out_channels)
            logweights (torch.Tensor): (bs, *, num_gaussians, h, w)
        gm_power (float): power for the Gaussian mixture, samples are approximated if power is not 1
        n_samples (int): number of samples
        cov_sharpen (bool): whether to sharpen the covariance matrix when power is greater than 1

    Returns:
        torch.Tensor: (bs, *, n_samples, out_channels, h, w)
    """
    gm_means = gm['means'].unsqueeze(-1).unsqueeze(-1)
    gm_logweights = gm['logweights'].unsqueeze(-1).unsqueeze(-1)

    if 'covs' in gm:
        gm_covs = gm['covs']

        batch_shapes = gm_means.shape[:-4]
        num_gaussians, h, w, out_channels = gm_means.shape[-4:]
        batch_numel = batch_shapes.numel()

        inds = torch.multinomial(
            (gm_logweights.reshape(batch_numel, num_gaussians, h, w).permute(0, 2, 3, 1).reshape(
                batch_numel * h * w, num_gaussians) * gm_power).softmax(dim=-1),
            n_samples, replacement=True
        ).reshape(batch_numel, h, w, n_samples).permute(0, 3, 1, 2).reshape(*batch_shapes, n_samples, 1, h, w)

        means = gm_means.gather(  # (bs, *, n_samples, h, w, out_channels)
            dim=-4,
            index=inds.reshape(*batch_shapes, n_samples, h, w, 1).expand(
                *batch_shapes, n_samples, h, w, out_channels))
        if gm_covs.size(-5) == 1:
            tril = torch.linalg.cholesky(gm_covs)  # (bs, *, 1, h, w, out_channels, out_channels)
        elif n_samples < num_gaussians:
            covs = gm_covs.gather(  # (bs, *, n_samples, h, w, out_channels, out_channels)
                dim=-5,
                index=inds.reshape(*batch_shapes, n_samples, h, w, 1, 1).expand(
                    *batch_shapes, n_samples, h, w, out_channels, out_channels))
            if cov_sharpen:
                covs = covs / gm_power
            tril = torch.linalg.cholesky(covs)
        else:
            tril = torch.linalg.cholesky(gm_covs)
            if cov_sharpen:
                tril = tril / math.sqrt(gm_power)
            tril = tril.gather(  # (bs, *, n_samples, h, w, out_channels, out_channels)
                dim=-5,
                index=inds.reshape(*batch_shapes, n_samples, h, w, 1, 1).expand(
                    *batch_shapes, n_samples, h, w, out_channels, out_channels))

        # (bs, *, n_samples, h, w, out_channels)
        samples = (tril @ torch.randn(
            (*batch_shapes, n_samples, h, w, out_channels, 1),
            dtype=means.dtype, device=means.device)).squeeze(-1) + means
        samples = samples.reshape(batch_numel, n_samples, h, w, out_channels).permute(0, 1, 4, 2, 3).reshape(
            *batch_shapes, n_samples, out_channels, h, w)

    else:
        batch_shapes = gm_means.shape[:-4]
        num_gaussians, out_channels, h, w = gm_means.shape[-4:]
        batch_numel = batch_shapes.numel()

        inds = torch.multinomial(
            (gm_logweights.reshape(batch_numel, num_gaussians, h, w).permute(0, 2, 3, 1).reshape(
                batch_numel * h * w, num_gaussians) * gm_power).softmax(dim=-1),
            n_samples, replacement=True
        ).reshape(batch_numel, h, w, n_samples).permute(0, 3, 1, 2).reshape(*batch_shapes, n_samples, 1, h, w)

        means = gm_means.gather(  # (bs, *, n_samples, out_channels, h, w)
            dim=-4,
            index=inds.expand(*batch_shapes, n_samples, out_channels, h, w))
        stds = gm['logstds'].exp().unsqueeze(-1).unsqueeze(-1)  # (bs, *, 1, 1, 1, 1) or (bs, *, num_gaussians, 1, h, w)
        if cov_sharpen:
            stds = stds / math.sqrt(gm_power)
        if stds.size(-4) == num_gaussians and num_gaussians > 1:
            stds = stds.gather(dim=-4, index=inds)  # (bs, *, n_samples, 1, h, w)

        # (bs, *, n_samples, out_channels, h, w)
        samples = stds * torch.randn(
            (*batch_shapes, n_samples, out_channels, h, w),
            dtype=means.dtype, device=means.device) + means
    return samples
class RectifiedFlow:
    def __init__(
        self,
        data_shape: tuple | list,
        velocity_field: nn.Module | Callable,
        interp: AffineInterp | str = "straight",
        source_distribution: (
            torch.distributions.Distribution | str | Callable
        ) = "normal",
        is_independent_coupling: bool = True,
        train_time_distribution: TrainTimeSampler | str = "uniform",
        train_time_weight: TrainTimeWeight | str = "uniform",
        criterion: RectifiedFlowLossFunction | str = "mse",
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
        *args,
        **kwargs
    ):
        r"""Initialize the RectifiedFlow class.

        Args:
            data_shape (`tuple`):
                Shape of the input data, excluding the batch dimension.
            velocity_field (`nn.Module`):
                Velocity field velocity_field that takes inputs `x_t` and time `t`, and outputs velocity `v_t`.
            interp (`AffineInterp` or `str`, *optional*, defaults to `"straight"`):
                Interpolation method for generating intermediate states.
                Can be an instance of `AffineInterp` or a string used to initialize the `AffineInterp` class.
            source_distribution (`torch.distributions.Distribution` or `str` or `Callable`, *optional*, defaults to `"normal"`):
                Source distribution `pi_0`. Can be:
                - A PyTorch distribution instance,
                - A callable for custom sampling,
                - A string `"normal"` for the standard Gaussian distribution.
            is_independent_coupling (`bool`, defaults to `True`):
                Indicates whether the rectified flow uses independent coupling.
                Set to `False` when using Reflow.
            train_time_distribution (`TrainTimeSampler` or `str`, *optional*, defaults to `"uniform"`):
                Distribution for sampling training times.
                Can be an instance of `TrainTimeSampler` or a string specifying the distribution type.
            train_time_weight (`TrainTimeWeight` or `str`, *optional*, defaults to `"uniform"`):
                Weight applied to training times.
                Can be an instance of `TrainTimeWeight` or a string specifying the weight type.
            criterion (`RectifiedFlowLossFunction` or `str`, *optional*, defaults to `"mse"`):
                Loss function used for training.
                Can be an instance of `RectifiedFlowLossFunction` or a string specifying the loss type.
        """
        self.data_shape = (data_shape,) if isinstance(data_shape, int) else tuple(data_shape)
        self.velocity_field = velocity_field

        self.interp: AffineInterp = (
            interp if isinstance(interp, AffineInterp) else AffineInterp(interp)
        )
        self.train_time_sampler: TrainTimeSampler = (
            train_time_distribution
            if isinstance(train_time_distribution, TrainTimeSampler)
            else TrainTimeSampler(train_time_distribution)
        )
        self.train_time_weight: TrainTimeWeight = (
            train_time_weight
            if isinstance(train_time_weight, TrainTimeWeight)
            else TrainTimeWeight(train_time_weight)
        )
        self.criterion: RectifiedFlowLossFunction = (
            criterion
            if isinstance(criterion, RectifiedFlowLossFunction)
            else RectifiedFlowLossFunction(criterion)
        )

        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = torch.dtype(dtype) if isinstance(dtype, str) else dtype

        self.pi_0 = source_distribution
        if self.pi_0 == "normal":
            self.pi_0 = dist.Normal(
                torch.tensor(0, device=device, dtype=dtype),
                torch.tensor(1, device=device, dtype=dtype),
            ).expand(data_shape)
        elif isinstance(self.pi_0, dist.Distribution):
            if (
                self.pi_0.mean.device != self.device
                or self.pi_0.stddev.device != self.device
            ):
                warnings.warn(
                    f"[Device Mismatch] The source distribution is on device "
                    f"{self.pi_0.mean.device}, while the model expects device {self.device}. "
                    f"Ensure that the distribution and model are on the same device."
                )
            if (
                self.pi_0.mean.dtype != self.dtype
                or self.pi_0.stddev.dtype != self.dtype
            ):
                warnings.warn(
                    f"[Dtype Mismatch] The source distribution uses dtype "
                    f"{self.pi_0.mean.dtype}, while the model expects dtype {self.dtype}. "
                    f"Consider converting the distribution to match the model's dtype."
                )

        self.independent_coupling = is_independent_coupling

    def sample_train_time(self, batch_size: int, expand_dim: bool = False):
        r"""This method calls the `TrainTimeSampler` to sample training times.

        Returns:
            t (`torch.Tensor`):
                A tensor of sampled training times with shape `(batch_size,)`,
                matching the class specified `device` and `dtype`.
        """
        time = self.train_time_sampler(batch_size, device=self.device, dtype=self.dtype)
        return self.match_dim_with_data(
            time, (batch_size, *self.data_shape), expand_dim=expand_dim
        )

    def sample_source_distribution(self, batch_size: int):
        r"""Sample data from the source distribution `pi_0`.

        Returns:
            x_0 (`torch.Tensor`):
                A tensor of sampled data with shape `(batch_size, *data_shape)`,
                matching the class specified `device` and `dtype`.
        """
        if True:
            return self.pi_0.sample((batch_size,)).to(self.device, self.dtype)
        elif callable(self.pi_0):
            return self.pi_0(batch_size).to(self.device, self.dtype)
        else:
            raise ValueError(
                "Source distribution must be a torch.distributions.Distribution or a callable."
            )

    def get_interpolation(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        t: torch.Tensor,
    ):
        r"""
        This method uses the interpolation method defined in `AffineInterp` to compute
        interpolation `X_t` and their time derivatives `dotX_t` at the specified time points `t`.

        Args:
            x_0 (`torch.Tensor`):
                shape `(B, D1, D2, ..., Dn)`, where `B` is the batch size, and `D1, D2, ..., Dn` are the data dimensions.
            x_1 (`torch.Tensor`):
                with the same shape as `X_0`
            t (`torch.Tensor`):
                A tensor of time steps, with shape `(B,)`, where each value is in `[0, 1]`.

        Returns:
            (x_t, dot_x_t) (`Tuple[torch.Tensor, torch.Tensor]`):
                - X_t (`torch.Tensor`): The interpolated state, with shape `(B, D1, D2, ..., Dn)`.
                - dotX_t (torch.Tensor): The time derivative of the interpolated state, with the same shape as `X_t`.
        """
        assert x_0.shape == x_1.shape, "x_0 and x_1 must have the same shape."
        assert x_0.shape[0] == x_1.shape[0], "Batch size of x_0 and x_1 must match."
        x_t, dot_x_t = self.interp.forward(x_0, x_1, t, detach=True)
        return x_t, dot_x_t

    def get_velocity(
        self,
        x_t: torch.Tensor | float | List[float],
        t: torch.Tensor,
        **kwargs,
    ):
        r"""
        This method calculates the velocity of the flow `v_t` at a given state `X_t` and time `t` using the provided model.

        Args:
            x_t (`torch.Tensor`):
                The state `X_t` at which to compute the velocity, with shape `(B, D_1, D_2, ..., D_n)`,
                where `B` is the batch size, and `D_1, D_2, ..., D_n` are the data dimensions.
            t (`torch.Tensor` | `float` | `List[float]`):
                Time tensor, which can be:
                - A scalar (`float` or 0-dimensional `torch.Tensor`)
                - A list of floats with length equal to the batch size or length 1
                - A `torch.Tensor` of shape `(B,)`, `(B, 1)`, or `(1,)`
            **kwargs:
                Additional keyword arguments to pass to the velocity field model.

        Returns:
            velocity (`torch.Tensor`):
                The velocity tensor `v_t`, with the same shape as `x_t` (`B, D_1, D_2, ..., D_n)`).
        """
        t = self.match_dim_with_data(t, x_t.shape, expand_dim=False).unsqueeze(-1)
        velocity = self.velocity_field(x_t, t, **kwargs)
        return velocity

    def get_loss(
        self,
        x_0: torch.Tensor | None,
        x_1: torch.Tensor,
        t: torch.Tensor | None = None,
        **kwargs,
    ):
        """Compute the loss of the rectified flow model, given samples `X_0, X_1`, and time `t`.

        This method calculates the loss by interpolating between the given data points `X_0` and `X_1`,
        computing the velocity field at the interpolated points, and comparing it to the time derivatives of the interpolation.
        The result is weighted by the specified time weight class and passed to the loss criterion.

        Args:
            x_0 (`torch.Tensor` or `None`):
                Samples from the source distribution `pi_0`, with shape `(B, D_1, D_2, ..., D_n)`,
                where `B` is the batch size and `D_1, D_2, ..., D_n` are the data dimensions.
                If `None`, samples are drawn from the source distribution `pi_0`.
            x_1 (`torch.Tensor`):
                Samples from the target distribution `pi_1`, with the same shape as `X_0`.
            t (`torch.Tensor` or `None`, *optional*, defaults to `None`):
                A tensor of time steps, with shape `(B,)`, where each value is in the range `[0, 1]`.
                If `None` or not provided, training times are sampled from the training time distribution.
            **kwargs:
                Additional keyword arguments passed to the velocity field model.

        Returns:
            loss (`torch.Tensor`):
                A scalar tensor representing the computed loss value.
        """
        t = self.sample_train_time(x_1.shape[0]) if t is None else t

        if x_0 is None:
            if self.is_independent_coupling:
                x_0 = self.sample_source_distribution(x_1.shape[0])
            else:
                warnings.warn(
                    "x_0 is not provided and is not independent coupling. Sampling from pi_0 might not be correct."
                )

        x_t, dot_x_t = self.get_interpolation(x_0, x_1, t)
        v_t = self.get_velocity(x_t, t, **kwargs)
        time_weights = self.train_time_weight(t)

        return self.criterion(
            v_t=v_t,
            dot_x_t=dot_x_t,
            x_t=x_t,
            t=t,
            time_weights=time_weights,
        )

    def get_score_function(self, x_t, t, **kwargs):
        r"""Compute the score function of the flow at a given `X_t` and time `t`.

        This method computes the score function `Dlogp_t(X_t)`, which represents the gradient of the log-probability
        of the current state with respect to the state `X_t`, using the model predicted velocity field `v_t`.

        Args:
            x_t (`torch.Tensor`):
                The state `X_t`, with shape `(B, D_1, D_2, ..., D_n)`, where `B` is the batch size.
            t (`torch.Tensor`):
                A tensor of time steps, with shape `(B,)`, where each value is in the range `[0, 1]`.
            **kwargs:
                Additional keyword arguments passed to the velocity field model.

        Returns:
            dlogp (`torch.Tensor`):
                The score function tensor, with the same shape as `x_t`.
        """
        v_t = self.get_velocity(x_t, t, **kwargs)
        return self.get_score_function_from_velocity(x_t, v_t, t)

    def get_score_function_from_velocity(self, x_t, v_t, t):
        r"""Compute the score function of the flow at `(X_t, t)`.

        This method calculates the score function `Dlogp_t(X_t)` using the given velocity field `v_t`,
        and then using the velocity field to calculate the score function.

        Note:
            The source distribution `pi_0` must be a Gaussian distribution.
            If `pi_0` is `N(0, I)`, the score function is calculated as: `Dlogp_t(X_t) = -E[X_0|X_t]/b_t`.
            For non-Gaussian `pi_0`, a custom score function must be provided.

        Args:
            x_t (`torch.Tensor`):
                The state `X_t`, with shape `(B, D_1, D_2, ..., D_n)`.
            v_t (`torch.Tensor`):
                The velocity field `v_t`, with the same shape as `x_t`.
            t (`torch.Tensor`):
                A tensor of time steps, with shape `(B,)`, where each value is in the range `[0, 1]`.

        Returns:
            dlogp (`torch.Tensor`):
                The score function tensor, with the same shape as `x_t`.

        Raises:
            Warning: If `pi_0` is not Gaussian or independent coupling is not used, a warning is issued.
        """
        if not self.independent_coupling or not self.is_pi_0_gaussian:
            warnings.warning(
                "The formula is theoretically correct only for independent couplings and Gaussian pi0, use at your own risk"
            )

        self.interp.solve(t=t, x_t=x_t, dot_x_t=v_t)
        dlogp = self.get_score_function_of_pi_0(self.interp.x_0) / self.interp.b_t
        return dlogp

    def get_score_function_of_pi_0(self, x_0):
        r"""Compute `Dlogp_0(X_0)`, the score function of the source distribution `pi_0` at `X_0`.

        Args:
            x_0 (`torch.Tensor`):
                Samples from the source distribution `pi_0`, with shape `(B, D_1, D_2, ..., D_n)`.

        Returns:
            dlopg (`torch.Tensor`):
                The score function tensor, with the same shape as `x_0`.
        """
        if self.is_pi_0_standard_gaussian:
            return -x_0
        elif isinstance(self.pi_0, torch.distributions.Normal):
            return -(
                x_0 - self.pi_0.mean.to(self.device, self.dtype)
            ) / self.pi_0.variance.to(self.device, self.dtype)
        elif isinstance(self.pi_0, torch.distributions.MultivariateNormal):
            return -(
                x_0 - self.pi_0.mean.to(self.device, self.dtype)
            ) @ self.pi_0.precision_matrix.to(self.device, self.dtype)
        else:
            try:
                return self.pi_0.score_function(x_0)
            except:
                raise ValueError(
                    "pi_0 is not a standard Gaussian distribution and must provide a score function."
                )

    def get_sde_params_by_sigma(self, v_t, x_t, t, sigma):
        # SDE coeffs for dX_t = v_t(X_t) + sigma_t^2*Dlogp(X_t) + sqrt(2)*sigma_t*dWt
        if not self.independent_coupling or not self.is_pi_0_gaussian:
            warnings.warning(
                "The formula is theoretically correct only for independent couplings and Gaussian pi0, use at your own risk"
            )
        sigma_t = sigma(t)
        result = self.interp.solve(t=t, x_t=x_t, dot_x_t=v_t)
        dlogp = -result.x_0 / result.b_t
        v_t_sde = v_t + sigma_t**2 * dlogp
        return v_t_sde, sigma_t * 2**0.5

    def get_stable_sde_params(self, v_t, x_t, t, e):
        # From SDE coeffs for dX = v_t(Xt) -sigma_t^2*E[X0|Xt]/bt + sqrt(2)*sigma_t*dWt,
        # let et^2 = sigmat^2/bt, we have sigmat = sqrt(bt) * et, we have:
        # dX = v_t(Xt) - et^2*E[X0|Xt]+ sqrt(2*bt) * et *dWt
        if not self.independent_coupling or not self.is_pi_0_gaussian:
            warnings.warning(
                "The formula is theoretically correct only for independent couplings and Gaussian pi0, use at your own risk"
            )
        result = self.interp.solve(t=t, x_t=x_t, dot_x_t=v_t)
        et = e(
            self.match_dim_with_data(t, x_t.shape, device=x_t.device, dtype=x_t.dtype)
        )
        x0_pred = -result.x_0 / result.b_t
        v_t_sde = v_t - x0_pred * et**2
        sigma_t = et * result.b_t**0.5 * (2**0.5)
        # at, bt, dot_at, dot_bt = self.interp.get_coeffs(t)
        # v_t_sde =v_t * (1+et) - et * dot_at / at * xt
        # sigma_t_sde = (2 * (1-at) * dot_at/(at) * et)**(0.5)
        return v_t_sde, sigma_t

    def match_dim_with_data(
        self,
        t: torch.Tensor | float | List[float],
        x_shape: tuple,
        expand_dim: bool = True,
    ):
        r"""Reshapes the time tensor `t` to match the dimensions of a tensor `X`.

        This is a wrapper for the standalone `match_dim_with_data` function, using the current class's
        device and dtype for tensor creation and operations.

        Args:
            t (`torch.Tensor` | `float` | `List[float]`):
                Time tensor, which can be:
                - A scalar (`float` or 0-dimensional `torch.Tensor`)
                - A list of floats with length equal to the batch size or length 1
                - A `torch.Tensor` of shape `(B,)`, `(B, 1)`, or `(1,)`
            x_shape (`tuple`):
                Shape of the tensor `X`, e.g., `X.shape`, used to determine the batch size and dimensions.
            expand_dim (`bool`, defaults to `True`):
                If `True`, reshapes `t` to include singleton dimensions after the batch dimension.

        Returns:
            `torch.Tensor`:
                Reshaped time tensor, matching the dimensions of the input tensor `X`.

        Example:
        >>> x_shape = (16, 3, 32, 32)
        >>> t_prepared = match_dim_with_data([0.5], x_shape, expand_dim=True)
        >>> t_prepared.shape
        torch.Size([16, 1, 1, 1])

        >>> t_prepared = match_dim_with_data([0.5], x_shape, expand_dim=False)
        >>> t_prepared.shape
        torch.Size([16])
        """
        return match_dim_with_data(
            t, x_shape, device=self.device, dtype=self.dtype, expand_dim=expand_dim
        )

    @property
    def is_pi_0_gaussian(self):
        """Check if pi_0 is a Gaussian distribution."""
        return isinstance(self.pi_0, dist.Normal) or isinstance(
            self.pi_0, dist.MultivariateNormal
        )

    @property
    def is_pi_0_zero_mean_gaussian(self):
        """Check if pi_0 is a zero-mean Gaussian distribution."""
        if callable(self.pi_0):
            warnings.warn("pi_0 is a custom distribution and may not have zero mean.")

        is_multivariate_normal = isinstance(
            self.pi_0, dist.MultivariateNormal
        ) and torch.allclose(self.pi_0.mean, torch.zeros_like(self.pi_0.mean))
        is_normal = isinstance(self.pi_0, dist.Normal) and torch.allclose(
            self.pi_0.loc, torch.zeros_like(self.pi_0.loc)
        )
        return is_multivariate_normal or is_normal

    @property
    def is_pi_0_standard_gaussian(self):
        """Check if pi_0 is a standard Gaussian distribution."""
        is_multivariate_normal = (
            isinstance(self.pi_0, dist.MultivariateNormal)
            and torch.allclose(self.pi_0.mean, torch.zeros_like(self.pi_0.mean))
            and torch.allclose(
                self.pi_0.covariance_matrix,
                torch.eye(self.pi_0.mean.size(0), device=self.pi_0.mean.device),
            )
        )
        is_normal = (
            isinstance(self.pi_0, dist.Normal)
            and torch.allclose(self.pi_0.mean, torch.zeros_like(self.pi_0.mean))
            and torch.allclose(self.pi_0.variance, torch.ones_like(self.pi_0.variance))
        )
        return is_multivariate_normal or is_normal

    @property
    def is_independent_coupling(self):
        """Check if rectified flow is a independent coupling."""
        return self.independent_coupling

    @property
    def is_canonical(self):
        """Check if the rectified flow is canonical
        i.e., pi_0 is a standard Gaussian and is the rectified flow is independent coupling.
        """
        return self.is_pi_0_standard_gaussian and self.is_independent_coupling



class ScoringRule(RectifiedFlow):
    
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.minibatch_size = 100
        self.beta = 2.0
        self.alpha = 1.0
        
        self.xi_distribution = dist.Normal(
            torch.zeros(self.data_shape, device=self.device, dtype=self.dtype),
            torch.ones(self.data_shape, device=self.device, dtype=self.dtype),
        )
    
    def get_velocity(
        self,
        x_t: torch.Tensor | float | List[float],
        t: torch.Tensor,
        **kwargs,
    ):
        r"""
        This method calculates the velocity of the flow `v_t` at a given state `X_t` and time `t` using the provided model.

        Args:
            x_t (`torch.Tensor`):
                The state `X_t` at which to compute the velocity, with shape `(B, D_1, D_2, ..., D_n)`,
                where `B` is the batch size, and `D_1, D_2, ..., D_n` are the data dimensions.
            t (`torch.Tensor` | `float` | `List[float]`):
                Time tensor, which can be:
                - A scalar (`float` or 0-dimensional `torch.Tensor`)
                - A list of floats with length equal to the batch size or length 1
                - A `torch.Tensor` of shape `(B,)`, `(B, 1)`, or `(1,)`
            **kwargs:
                Additional keyword arguments to pass to the velocity field model.

        Returns:
            velocity (`torch.Tensor`):
                The velocity tensor `v_t`, with the same shape as `x_t` (`B, D_1, D_2, ..., D_n)`).
        """
        
        t = self.match_dim_with_data(t, x_t.shape, expand_dim=False).unsqueeze(-1)
        xi = self.xi_distribution.sample((x_t.shape[0],))
        x1_hat = self.velocity_field(x_t, t, xi)
        velocity = (x1_hat - x_t) / (1 - t)
        # velocity = self.velocity_field(x_t, t, **kwargs)
        return velocity
        
    
    def sample_posterior(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        xi: torch.Tensor,
    ):
        temp_data_shape = (x_t.shape[0], x_t.shape[1])
        flat_data_dim= x_t.shape[0] * x_t.shape[1]
        x_1_hat = self.velocity_field(x_t.view(flat_data_dim, *self.data_shape), t.view(flat_data_dim, 1), xi.view(flat_data_dim, *self.data_shape)).view(*temp_data_shape, *self.data_shape)
        return x_1_hat

    
    
    def get_loss(
        self,
        x_0: torch.Tensor | None,
        x_1: torch.Tensor,
        t: torch.Tensor | None = None,
    ):
        """Compute the loss of the rectified flow model, given samples `X_0, X_1`, and time `t`.

        This method calculates the loss by interpolating between the given data points `X_0` and `X_1`,
        computing the velocity field at the interpolated points, and comparing it to the time derivatives of the interpolation.
        The result is weighted by the specified time weight class and passed to the loss criterion.

        Args:
            x_0 (`torch.Tensor` or `None`):
                Samples from the source distribution `pi_0`, with shape `(B, D_1, D_2, ..., D_n)`,
                where `B` is the batch size and `D_1, D_2, ..., D_n` are the data dimensions.
                If `None`, samples are drawn from the source distribution `pi_0`.
            x_1 (`torch.Tensor`):
                Samples from the target distribution `pi_1`, with the same shape as `X_0`.
            t (`torch.Tensor` or `None`, *optional*, defaults to `None`):
                A tensor of time steps, with shape `(B,)`, where each value is in the range `[0, 1]`.
                If `None` or not provided, training times are sampled from the training time distribution.
            **kwargs:
                Additional keyword arguments passed to the velocity field model.

        Returns:
            loss (`torch.Tensor`):
                A scalar tensor representing the computed loss value.
        """
        t = self.sample_train_time(x_1.shape[0]) if t is None else t

        if x_0 is None:
            if self.is_independent_coupling:
                x_0 = self.sample_source_distribution(x_1.shape[0])
            else:
                warnings.warn(
                    "x_0 is not provided and is not independent coupling. Sampling from pi_0 might not be correct."
                )
                
        # sample xi
        xi = self.xi_distribution.sample((x_1.shape[0], self.minibatch_size)).to(
            self.device, self.dtype
        )

                

        x_t, dot_x_t = self.get_interpolation(x_0, x_1, t)
        
        x_t = x_t.unsqueeze(1).repeat(1, self.minibatch_size, 1)
        t = t.unsqueeze(1).repeat(1, self.minibatch_size).unsqueeze(-1)
        x_1 = x_1.unsqueeze(1).repeat(1, self.minibatch_size, 1)
        x_1_hat = self.sample_posterior(x_t, t, xi)
        # time_weights = self.train_time_weight(t)

        # calculate scoring rule loss
        l_data = torch.sqrt(1e-6 + torch.sum((x_1_hat - x_1)**2, dim=-1))**self.beta
        l_xi = torch.sqrt(1e-6 + torch.sum((x_1_hat.unsqueeze(2) - x_1_hat.unsqueeze(1))**2, dim=-1))**self.beta

        loss = torch.mean(l_data - self.alpha * torch.sum(l_xi, dim=-1) / 2 / (self.minibatch_size - 1))

        return loss

        
        
class GMFlow(RectifiedFlow):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.T_SCALE = 1000
    
    def gm_kl_loss(self, gm, sample, eps=1e-4):
        """
        Gaussian mixture KL divergence loss (without constant terms), a.k.a. GM NLL loss.

        Args:
            gm (dict):
                means (torch.Tensor): (bs, num_gaussians, D)
                logstds (torch.Tensor): (bs, 1, 1)
                logweights (torch.Tensor): (bs, num_gaussians, 1)
            sample (torch.Tensor): (bs, D)

        Returns:
            torch.Tensor: (bs, )
        """
        means = gm['means']
        logstds = gm['logstds']
        logweights = gm['logweights']

        inverse_stds = torch.exp(-logstds).clamp(max=1 / eps)
        diff_weighted = (sample.unsqueeze(-2) - means) * inverse_stds  # (bs, num_gaussians, D)
        gaussian_ll = (-0.5 * diff_weighted.square() - logstds).sum(dim=-1)  # (bs, num_gaussians)
        gm_nll = -torch.logsumexp(gaussian_ll + logweights.squeeze(-1) + 1e-9, dim=-1)  # (bs, )
        return gm_nll
    
    
    
    def get_loss(
        self,
        x_0: torch.Tensor | None,
        x_1: torch.Tensor,
        t: torch.Tensor | None = None,
    ):
        t = self.sample_train_time(x_1.shape[0]) if t is None else t
        if x_0 is None:
            if self.is_independent_coupling:
                x_0 = self.sample_source_distribution(x_1.shape[0])
            else:
                warnings.warn(
                    "x_0 is not provided and is not independent coupling. Sampling from pi_0 might not be correct."
                )
        
        x_t, dot_x_t = self.get_interpolation(x_0, x_1, t)
        gm_pred = self.velocity_field(x_t, t * self.T_SCALE)
        
        loss = self.gm_kl_loss(gm_pred, x_1)
        loss = loss.mean()
        return loss
    
    def get_velocity(
        self,
        x_t: torch.Tensor | float | List[float],
        t: torch.Tensor,
        **kwargs,
    ):
        r"""
        This method calculates the velocity of the flow `v_t` at a given state `X_t` and time `t` using the provided model.

        Args:
            x_t (`torch.Tensor`):
                The state `X_t` at which to compute the velocity, with shape `(B, D_1, D_2, ..., D_n)`,
                where `B` is the batch size, and `D_1, D_2, ..., D_n` are the data dimensions.
            t (`torch.Tensor` | `float` | `List[float]`):
                Time tensor, which can be:
                - A scalar (`float` or 0-dimensional `torch.Tensor`)
                - A list of floats with length equal to the batch size or length 1
                - A `torch.Tensor` of shape `(B,)`, `(B, 1)`, or `(1,)`
            **kwargs:
                Additional keyword arguments to pass to the velocity field model.

        Returns:
            velocity (`torch.Tensor`):
                The velocity tensor `v_t`, with the same shape as `x_t` (`B, D_1, D_2, ..., D_n)`).
        """
        t = self.match_dim_with_data(t, x_t.shape, expand_dim=False)
        t = t * self.T_SCALE
        
        gm_pred = self.velocity_field(x_t, t)
        x1_hat = gm_to_sample(gm_pred).reshape(x_t.shape[0], *self.data_shape)
        velocity = (x1_hat - x_t) / (1 - t.unsqueeze(-1) / self.T_SCALE)
        return velocity
    
    
class HRF(RectifiedFlow):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.depth = kwargs.get('depth', 3)
        if self.depth == 3:
            self.N_list = [2, 5, 10]
        elif self.depth == 2:
            self.N_list = [2, 50]
        self.A = torch.tril(torch.ones((self.depth, self.depth),device=self.device),diagonal=-1)
    
    
    def get_loss(
        self,
        x_0: torch.Tensor | None,
        x_1: torch.Tensor,
        t: torch.Tensor | None = None,
    ):
        """Compute the loss of the rectified flow model, given samples `X_0, X_1`, and time `t`.

        This method calculates the loss by interpolating between the given data points `X_0` and `X_1`,
        computing the velocity field at the interpolated points, and comparing it to the time derivatives of the interpolation.
        The result is weighted by the specified time weight class and passed to the loss criterion.

        Args:
            x_0 (`torch.Tensor` or `None`):
                Samples from the source distribution `pi_0`, with shape `(B, D_1, D_2, ..., D_n)`,
                where `B` is the batch size and `D_1, D_2, ..., D_n` are the data dimensions.
                If `None`, samples are drawn from the source distribution `pi_0`.
            x_1 (`torch.Tensor`):
                Samples from the target distribution `pi_1`, with the same shape as `X_0`.
            t (`torch.Tensor` or `None`, *optional*, defaults to `None`):
                A tensor of time steps, with shape `(B,)`, where each value is in the range `[0, 1]`.
                If `None` or not provided, training times are sampled from the training time distribution.
            **kwargs:
                Additional keyword arguments passed to the velocity field model.

        Returns:
            loss (`torch.Tensor`):
                A scalar tensor representing the computed loss value.
        """
        # t = self.sample_train_time(x_1.shape[0]) if t is None else t

        # if x_0 is None:
        #     if self.is_independent_coupling:
        #         x_0 = self.sample_source_distribution(x_1.shape[0])
        #     else:
        #         warnings.warn(
        #             "x_0 is not provided and is not independent coupling. Sampling from pi_0 might not be correct."
        #         )

        # x_t, dot_x_t = self.get_interpolation(x_0, x_1, t)

        # A = torch.tril(torch.ones((self.depth, self.depth),device=self.device),diagonal=-1)

        x_0 = torch.cat([x_0[:,None,:], torch.randn((x_0.shape[0], self.depth - 1) + x_0.shape[1:], device=self.device)], dim=1)   # N x D x d
        t = torch.rand((x_1.shape[0], self.depth)+(1,)*(x_1.dim()-1), device=self.device)
        x_t = (1-t)*x_0 + t*(x_1[:,None,...] - torch.einsum('ij,bj...->bi...', self.A, x_0))
        pred = self.velocity_field(x_t, t.squeeze(2))
        target = x_1 - torch.sum(x_0, dim=1) # N x d
        pred = self.velocity_field(x_t, t)
        loss = torch.mean((target - pred) ** 2)
        return loss
    
    @torch.no_grad()
    def sample_hierarchical(self, x_t, t, cur_depth=0, return_traj=False):
        x_0 = x_t[:,cur_depth,...].clone()
        local_num_steps = self.N_list[cur_depth]
        times = torch.linspace(0.0,1.0,local_num_steps+1,device=x_t.device)
        dts = torch.diff(times)
        if return_traj:
            traj = [x_0]
        for k in range(local_num_steps):
            current_time = times[k]
            dt = dts[k]
            t[cur_depth] = current_time
            if k == 0 and cur_depth != 0:
                x_0 = torch.randn_like(x_t[:,cur_depth,...],device=x_t.device)
                x_t[:,cur_depth,...] = x_0
            if cur_depth+1==self.depth:
                f = self.velocity_field(x_t,t*torch.ones((x_t.shape[0],1),device=x_t.device))
                x_t[:,cur_depth,...] += dt*f
            else:
                x_t[:,cur_depth,...] += dt*self.sample_hierarchical(x_t, t, cur_depth=cur_depth+1)[1]
            if return_traj:
                traj.append(x_t[:,cur_depth,...].detach().clone())
        if return_traj:
            return x_0, x_t[:,cur_depth,...], torch.stack(traj)
        else:
            return x_0, x_t[:,cur_depth,...]
        
        
    def get_velocity(self, x_t, t, **kwargs):
        
        x_t = torch.cat([x_t[:,None,:], torch.randn((x_t.shape[0], self.depth - 1) + x_t.shape[1:], device=self.device)], dim=1)   # N x D x d

        # t = self.match_dim_with_data(t, x_t.shape, expand_dim=False).unsqueeze(-1).repeat(1, self.depth)
        ts_all = torch.zeros(self.depth, device=self.device)
        ts_all[0] = t
        
        return self.sample_hierarchical(x_t, ts_all, cur_depth=1, return_traj=False)[1]