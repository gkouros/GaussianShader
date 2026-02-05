#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from matplotlib import cm
import cv2
import numpy as np

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def apply_colormap(image, cmap="viridis"):
    colormap = cm.get_cmap(cmap)
    colormap = torch.tensor(colormap.colors).to(image.device)  # type: ignore
    image_long = (image * 255).long()
    image_long_min = torch.min(image_long)
    image_long_max = torch.max(image_long)
    assert image_long_min >= 0, f"the min value is {image_long_min}"
    assert image_long_max <= 255, f"the max value is {image_long_max}"
    return colormap[image_long[..., 0]]

def apply_depth_colormap(depth, cmap="turbo", min=None, max=None):
    near_plane = float(torch.min(depth)) if min is None else min
    far_plane = float(torch.max(depth)) if max is None else max

    depth = (depth - near_plane) / (far_plane - near_plane + 1e-10)
    depth = torch.clip(depth, 0, 1)

    colored_image = apply_colormap(depth, cmap=cmap)
    return colored_image

def erode(img_in, erode_size=4):
    img_out = np.copy(img_in)
    kernel = np.ones((erode_size, erode_size), np.uint8)
    img_out = cv2.erode(img_out, kernel, iterations=1)

    return img_out

# sRGB ↔ linear transfer                              IEC 61966-2-1
_BREAK_FWD  = 0.0031308   # linear-to-sRGB  breakpoint (≈12.92⁻¹ * 0.04045)
_BREAK_INV  = 0.04045     # sRGB-to-linear breakpoint (IEC spec)
_SCALE_FWD  = 12.92       # slope in the linear segment (forward)
_SCALE_INV  = 1.0 / _SCALE_FWD
_A          = 1.055       # scale for power segment
_GAMMA      = 2.4         # exponent for power segment

def linear_to_srgb(linear, eps=None):
    """
    Convert linear-light RGB → sRGB.
    Accepts NumPy ndarray or Torch tensor, in [0,1].
    """
    mod, where, power, eps_default = _backend(linear)
    if eps is None:
        eps = eps_default
    # Two branches: linear segment vs power-law segment
    srgb = where(
        linear <= _BREAK_FWD,
        _SCALE_FWD * linear,
        _A * power(mod.clip(linear, eps, None), 1.0 / _GAMMA) - (_A - 1.0),
    )
    return srgb

def srgb_to_linear(srgb, eps=None):
    """
    Convert sRGB → linear-light RGB.
    Accepts NumPy ndarray or Torch tensor, in [0,1].
    """
    mod, where, power, eps_default = _backend(srgb)
    if eps is None:
        eps = eps_default
    linear = where(
        srgb <= _BREAK_INV,
        _SCALE_INV * srgb,
        power(mod.clip((srgb + (_A - 1.0)), eps, None) / _A, _GAMMA),
    )
    return linear
