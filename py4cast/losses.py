"""
This module contains the loss functions used in the training of the models.
"""

from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Tuple

import lightning.pytorch as pl
import torch
from torch.nn import MSELoss

from py4cast.datasets.base import DatasetInfo, NamedTensor


class AlmostFairCRPS(torch.nn.Module): # Batched Almost Fair CRPS
    """
    AIFS-CRPS: Ensemble forecasting using a model trained with a loss function based on the Continuous Ranked Probability Score
    https://arxiv.org/abs/2412.15832
    """
    def __init__(self, alpha=0.95):
        super().__init__()
        self.alpha = alpha

    def forward(self, preds, targets, noise_members=2):

        B = targets.shape[0] # effective batch size 
        M = noise_members  # Number of ensemble members
        epsilon = (1 - self.alpha) / M

        # Reshape predictions 
        preds = preds.reshape(B, M, *preds.shape[1:])        # (B, M, T, W, H, d_f)
        # Replicate targets along the members dimension
        targets = targets.unsqueeze(1).expand(-1, M, -1, -1, -1, -1)  # (B, M, T, W, H, d_f)

        abs_diff = torch.abs(preds - targets)             # (B, M, T, W, H, d_f)

        # Pairwise differences: |x_j - x_k|
        preds_j = preds.unsqueeze(2)                      # (B, M, 1, T, W, H, d_f)
        preds_k = preds.unsqueeze(1)                      # (B, 1, M, T, W, H, d_f)
        pairwise_diff = torch.abs(preds_j - preds_k)      # (B, M, M, T, W, H, d_f)

        # |x_j - y| + |x_k - y|
        abs_j = abs_diff.unsqueeze(2)                     # (B, M, 1, T, W, H, d_f)
        abs_k = abs_diff.unsqueeze(1)                     # (B, 1, M, T, W, H, d_f)
        pairwise_abs_sum = abs_j + abs_k                  # (B, M, M, T, W, H, d_f)

        # Create off-diagonal mask (M, M)
        mask = ~torch.eye(M, dtype=torch.bool, device=preds.device)  # (M, M)

        # Apply mask per batch element
        pairwise_diff = pairwise_diff[:, mask].view(B, M, M - 1, *preds.shape[2:])
        pairwise_abs_sum = pairwise_abs_sum[:, mask].view(B, M, M - 1, *preds.shape[2:])

        # Final computation
        afcrps = (pairwise_abs_sum - (1 - epsilon) * pairwise_diff).mean(dim=(1, 2)) / 2  # (B, T, W, H, d_f)
        return afcrps


class Py4CastLoss(ABC):
    """
    Abstract class to force the user to implement the prepare and forward method because
    They are expected by the rest of the system.
    See https://lightning.ai/docs/pytorch/stable/accelerators/accelerator_prepare.html
    """

    def __init__(self, loss: str, *args, **kwargs) -> None:
        if loss in ["MSELoss", "L1Loss"]:
            self.loss = getattr(torch.nn, loss)(*args, **kwargs)
        elif loss == "AFCRPS":
            self.loss = AlmostFairCRPS()
        else:
            raise ValueError("Unrecognized loss function")

    @abstractmethod
    def prepare(
        self,
        lm: pl.LightningModule,
        interior_mask: torch.Tensor,
        dataset_info: DatasetInfo,
    ) -> None:
        """
        Prepare the loss function using the dataset informations and the interior mask
        """

    @abstractmethod
    def forward(
        self, prediction: NamedTensor, target: NamedTensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the loss function
        """

    def register_loss_state_buffers(
        self,
        lm: pl.LightningModule,
        interior_mask: torch.Tensor,
        loss_state_weight: dict,
        squeeze_mask: bool = False,
    ) -> None:
        """
        We register the state_weight buffer to the lightning module
        and keep references to other buffers of interest
        """

        self.loss_state_weight = loss_state_weight
        attr_name = "interior_mask_s" if squeeze_mask else "interior_mask"
        if not hasattr(lm, attr_name):
            lm.register_buffer(
                attr_name,
                interior_mask.squeeze(-1) if squeeze_mask else interior_mask,
                persistent=False,
            )
        self.num_interior = torch.sum(interior_mask).item()

    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)

    @lru_cache(maxsize=8)
    def weights(self, feature_names: Tuple[str], device: torch.device) -> torch.Tensor:
        """
        We build and cache the weights tensor for the given loss, feature names and device.
        """
        return torch.stack([self.loss_state_weight[name] for name in feature_names]).to(
            device, non_blocking=True
        )


class WeightedLoss(Py4CastLoss):
    """
    Compute a weighted loss function with a weight for each feature.
    During the forward step, the loss is computed for each feature and then weighted
    and optionally averaged over the spatial dimensions.
    """

    def prepare(
        self,
        lm: pl.LightningModule,
        interior_mask: torch.Tensor,
        dataset_info: DatasetInfo,
    ) -> None:
        # build the dictionnary of weight
        loss_state_weight = {}

        for name in dataset_info.state_weights:
            loss_state_weight[name] = torch.tensor(dataset_info.state_weights[name])
            
        self.register_loss_state_buffers(
            lm, interior_mask, loss_state_weight, squeeze_mask=True
        )
        self.lm = lm

    def forward(
        self,
        prediction: NamedTensor,
        target: NamedTensor,
        mask: torch.Tensor,
        noise_members: int = 2,
        reduce_spatial_dim=True,
    ) -> torch.Tensor:
        """
        Computed weighted loss function.
        prediction/target: (B, pred_steps, N_grid, d_f) or (B, pred_steps, W, H, d_f)
        returns (B, pred_steps)
        """
        # Compute Torch loss (defined in the parent class when this Mixin is used)
        if noise_members > 1:
            torch_loss = self.loss(prediction.tensor * mask.repeat(noise_members, 1, 1, 1, 1) , target.tensor * mask, noise_members=noise_members)
        else:
            if self.loss.__class__ == AlmostFairCRPS:
                raise ValueError("When using AFCRPS loss, noise_members must be > 1")
            torch_loss = self.loss(prediction.tensor * mask, target.tensor * mask)

        # Retrieve the weights for each feature
        weights = self.weights(tuple(prediction.feature_names), prediction.device)

        # Apply the weights and sum over the feature dimension
        weighted_loss = torch.sum(torch_loss * weights, dim=-1)

        # if no reduction on spatial dimension is required, return the weighted loss
        if not reduce_spatial_dim:
            return weighted_loss

        union_mask = torch.any(mask, dim=(0, 1, 4))

        # Compute the mean loss over all spatial dimensions
        # Take (unweighted) mean over only non-border (interior) grid nodes/pixels
        # We use forward indexing for the spatial_dim_idx of the target tensor
        # so the code below works even if the feature dimension has been reduced
        # The final shape is (B, pred_steps)

        time_step_mean_loss = torch.sum(
            weighted_loss * self.lm.interior_mask_s,
            dim=target.spatial_dim_idx,
        ) / (self.num_interior - (~union_mask).sum())

        return time_step_mean_loss


class ScaledLoss(Py4CastLoss):
    def prepare(
        self,
        lm: pl.LightningModule,
        interior_mask: torch.Tensor,
        dataset_info: DatasetInfo,
    ) -> None:
        # build the dictionnary of weight
        loss_state_weight = {}
        for name in dataset_info.state_weights:
            loss_state_weight[name] = dataset_info.stats[name]["std"]
        self.register_loss_state_buffers(lm, interior_mask, loss_state_weight)
        self.lm = lm

    def forward(
        self, prediction: NamedTensor, target: NamedTensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Computed weighted loss function averaged over all spatial dimensions.
        prediction/target: (B, pred_steps, N_grid, d_f) or (B, pred_steps, W, H, d_f)
        returns (B, pred_steps)
        """
        # Compute Torch loss (defined in the parent class when this Mixin is used)
        torch_loss = self.loss(prediction.tensor * mask, target.tensor * mask)

        union_mask = torch.any(mask, dim=(0, 1, 4))

        # Compute the mean loss value over spatial dimensions
        mean_loss = torch.sum(
            torch_loss * self.lm.interior_mask,
            dim=target.spatial_dim_idx,
        ) / (self.num_interior - (~union_mask).sum())

        if self.loss.__class__ == MSELoss:
            mean_loss = torch.sqrt(mean_loss)

        return mean_loss * self.weights(
            tuple(prediction.feature_names), prediction.device
        )
