"""
Copyright (2025) Bytedance Ltd. and/or its affiliates
SPDX-License-Identifier: Apache-2.0
"""


import numpy as np
import torch 


# def exp(a, b):
#     return (a + b + np.pi) % (2 * np.pi) - np.pi

# def log(a, b):
#     return torch.atan2(torch.sin(b - a), torch.cos(b - a))

def interpolate(a, b, t):
    # return exp(a, t * log(a, b))
    return (a + t * wrap(b - a)) % (2 * np.pi)

def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def mod_to_standard_angle_range(a):
    return a % (2 * np.pi)

def vf(a, b, t):
    return wrap(b - a)

def vf2(a, b, t):
    c = interpolate(a, b, t)
    return wrap((b - c) / (1 - t))

def calc_torus_vf(x1, xt, t):
    # return wrap((x1 - xt) / (1 - t))
    # return (x1 - xt) / (1 - t)
    return wrap(x1 - xt) / (1 - t)
    



if __name__ == "__main__":
    a = torch.randn(20) * 2 * np.pi
    b = torch.randn(20) * 2 * np.pi
    t = 0.5
    v1 = vf(a, b, t)
    v2 = vf2(a, b, t)
    