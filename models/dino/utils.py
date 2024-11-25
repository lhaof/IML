# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import torch
from torch import nn, Tensor

import math
import torch.nn.functional as F
from torch import nn

import numpy as np
from util.misc import (NestedTensor, nested_tensor_from_tensor_list)


def gen_encoder_output_proposals(memory:Tensor, memory_padding_mask:Tensor, spatial_shapes:Tensor, learnedwh=None):
    """
    Input:
        - memory: bs, \sum{hw}, d_model
        - memory_padding_mask: bs, \sum{hw}
        - spatial_shapes: nlevel, 2
        - learnedwh: 2
    Output:
        - output_memory: bs, \sum{hw}, d_model
        - output_proposals: bs, \sum{hw}, 4
    """
    N_, S_, C_ = memory.shape
    base_scale = 4.0
    proposals = []
    _cur = 0
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
        valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
        valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

        grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                        torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
        grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1) # H_, W_, 2

        scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
        grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale

        if learnedwh is not None:
            wh = torch.ones_like(grid) * learnedwh.sigmoid() * (2.0 ** lvl)
        else:
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)

        proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
        proposals.append(proposal)
        _cur += (H_ * W_)

    output_proposals = torch.cat(proposals, 1)
    output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
    output_proposals = torch.log(output_proposals / (1 - output_proposals)) # unsigmoid
    output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
    output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

    output_memory = memory
    output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
    output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))

    return output_memory, output_proposals


class RandomBoxPerturber():
    def __init__(self, x_noise_scale=0.2, y_noise_scale=0.2, w_noise_scale=0.2, h_noise_scale=0.2) -> None:
        self.noise_scale = torch.Tensor([x_noise_scale, y_noise_scale, w_noise_scale, h_noise_scale])

    def __call__(self, refanchors: Tensor) -> Tensor:
        nq, bs, query_dim = refanchors.shape
        device = refanchors.device

        noise_raw = torch.rand_like(refanchors)
        noise_scale = self.noise_scale.to(device)[:query_dim]

        new_refanchors = refanchors * (1 + (noise_raw - 0.5) * noise_scale)
        return new_refanchors.clamp_(0, 1)


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def _get_activation_fn(activation, d_model=256, batch_dim=0):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "prelu":
        return nn.PReLU()
    if activation == "selu":
        return F.selu

    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def gen_sineembed_for_position(pos_tensor):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos
    
    
####
def xentropy_loss(true, pred, reduction="mean"):
    """Cross entropy loss. Assumes NHWC!

    Args:
        pred: prediction array
        true: ground truth array
    
    Returns:
        cross entropy loss

    """

    #print(true.shape,pred.shape)
    
    epsilon = 10e-8
    # scale preds so that the class probs of each sample sum to 1
    pred = pred / torch.sum(pred, -1, keepdim=True)
    # manual computation of crossentropy
    pred = torch.clamp(pred, epsilon, 1.0 - epsilon)
    loss = -torch.sum((true * torch.log(pred)), -1, keepdim=True)
    loss = loss.mean() if reduction == "mean" else loss.sum()
    return loss


####
def dice_loss(true, pred, smooth=1e-3):
    """`pred` and `true` must be of torch.float32. Assuming of shape NxHxWxC."""
    
    
    inse = torch.sum(pred * true, (0, 1, 2))
    l = torch.sum(pred, (0, 1, 2))
    r = torch.sum(true, (0, 1, 2))
    loss = 1.0 - (2.0 * inse + smooth) / (l + r + smooth)
    loss = torch.sum(loss)
    return loss


####
def mse_loss(true, pred):
    """Calculate mean squared error loss.

    Args:
        true: ground truth of combined horizontal
              and vertical maps
        pred: prediction of combined horizontal
              and vertical maps 
    
    Returns:
        loss: mean squared error

    """
    loss = pred - true
    loss = (loss * loss).mean()
    return loss


####
def msge_loss(true, pred, focus):
    """Calculate the mean squared error of the gradients of 
    horizontal and vertical map predictions. Assumes 
    channel 0 is Vertical and channel 1 is Horizontal.

    Args:
        true:  ground truth of combined horizontal
               and vertical maps
        pred:  prediction of combined horizontal
               and vertical maps 
        focus: area where to apply loss (we only calculate
                the loss within the nuclei)
    
    Returns:
        loss:  mean squared error of gradients

    """

    def get_sobel_kernel(size):
        """Get sobel kernel with a given size."""
        assert size % 2 == 1, "Must be odd, get size=%d" % size

        h_range = torch.arange(
            -size // 2 + 1,
            size // 2 + 1,
            dtype=torch.float32,
            device="cuda",
            requires_grad=False,
        )
        v_range = torch.arange(
            -size // 2 + 1,
            size // 2 + 1,
            dtype=torch.float32,
            device="cuda",
            requires_grad=False,
        )
        h, v = torch.meshgrid(h_range, v_range)
        kernel_h = h / (h * h + v * v + 1.0e-15)
        kernel_v = v / (h * h + v * v + 1.0e-15)
        return kernel_h, kernel_v

    ####
    def get_gradient_hv(hv):
        """For calculating gradient."""
        kernel_h, kernel_v = get_sobel_kernel(5)
        kernel_h = kernel_h.view(1, 1, 5, 5)  # constant
        kernel_v = kernel_v.view(1, 1, 5, 5)  # constant

        h_ch = hv[..., 0].unsqueeze(1)  # Nx1xHxW
        v_ch = hv[..., 1].unsqueeze(1)  # Nx1xHxW

        # can only apply in NCHW mode
        h_dh_ch = F.conv2d(h_ch, kernel_h, padding=2)
        v_dv_ch = F.conv2d(v_ch, kernel_v, padding=2)
        dhv = torch.cat([h_dh_ch, v_dv_ch], dim=1)
        dhv = dhv.permute(0, 2, 3, 1).contiguous()  # to NHWC
        return dhv

    focus = (focus[..., None]).float()  # assume input NHW
    focus = torch.cat([focus, focus], axis=-1)
    true_grad = get_gradient_hv(true)
    pred_grad = get_gradient_hv(pred)
    loss = pred_grad - true_grad
    loss = focus * (loss * loss)
    # artificial reduce_mean with focused region
    loss = loss.sum() / (focus.sum() + 1.0e-8)
    return loss