"""Cautious-SYRE kernel math.

The cautious mask gates the SYRE diff element-wise by ``(U * theta >=
0)``, where ``U`` is the update tensor the optimizer is about to
subtract from theta. These tests pin the mask semantics: matches a
hand-rolled reference for a random U, reduces to basic SYRE when the
mask is all-true (``U = theta`` -> ``U*theta = theta**2 >= 0``), and
no-ops entirely when the mask is all-false (``U = -theta``).
"""

from __future__ import annotations

import importlib.util

import pytest
import torch


CUDA_AVAILABLE = torch.cuda.is_available()
TRITON_AVAILABLE = importlib.util.find_spec("triton") is not None
HAS_TRITON_GPU = CUDA_AVAILABLE and TRITON_AVAILABLE

gpu_only = pytest.mark.skipif(
    not HAS_TRITON_GPU, reason="SYRE requires CUDA + triton"
)


@gpu_only
def test_cautious_syre_matches_hand_rolled_reference():
    """Compute ``theta_0`` by running plain SYRE with ``gamma=1`` from
    ``theta=0``, then hand-roll the cautious-SYRE update in PyTorch and
    compare to the kernel's output.
    """
    from dion.syre import syre_wd_inplace

    torch.manual_seed(2026)
    n = 1024
    theta = torch.randn(n, device="cuda") * 0.5
    theta_orig = theta.clone()
    U = torch.randn(n, device="cuda")

    seed1 = 17
    std = 0.1
    gamma = 0.3

    theta_zero = torch.zeros(n, device="cuda")
    syre_wd_inplace(
        theta_zero, gamma=1.0, seed1=seed1, std=std,
        seed2=0, d_bound=0.0, advanced_removal=False, offset_base=0,
    )
    theta_0 = theta_zero

    mask = (U * theta_orig >= 0).to(theta_orig.dtype)
    diff = theta_orig - theta_0
    expected = theta_orig - gamma * diff * mask

    theta_kernel = theta_orig.clone()
    syre_wd_inplace(
        theta_kernel, gamma=gamma, seed1=seed1, std=std,
        seed2=0, d_bound=0.0, advanced_removal=False, offset_base=0,
        U=U,
    )
    torch.testing.assert_close(theta_kernel, expected, atol=1e-5, rtol=1e-5)


@gpu_only
def test_cautious_syre_all_true_mask_matches_basic():
    """``U = theta`` -> ``U * theta = theta**2 >= 0`` everywhere, so
    cautious-SYRE reduces to basic SYRE (to fp32 ULP).
    """
    from dion.syre import syre_wd_inplace

    torch.manual_seed(11)
    n = 512
    theta = torch.randn(n, device="cuda") * 0.5
    theta_basic = theta.clone()
    theta_cautious = theta.clone()
    U = theta.clone()
    syre_wd_inplace(
        theta_basic, gamma=0.4, seed1=99, std=0.1,
        seed2=0, d_bound=0.0, advanced_removal=False, offset_base=0,
    )
    syre_wd_inplace(
        theta_cautious, gamma=0.4, seed1=99, std=0.1,
        seed2=0, d_bound=0.0, advanced_removal=False, offset_base=0,
        U=U,
    )
    torch.testing.assert_close(theta_cautious, theta_basic, atol=1e-7, rtol=1e-6)


@gpu_only
def test_cautious_syre_all_false_mask_is_noop():
    """``U = -theta`` (and theta strictly nonzero) -> mask is 0
    everywhere, so the cautious kernel leaves theta untouched.
    """
    from dion.syre import syre_wd_inplace

    torch.manual_seed(13)
    n = 512
    theta = (torch.randn(n, device="cuda") * 0.5).abs() + 0.01
    theta = torch.where(
        torch.rand(n, device="cuda") < 0.5, -theta, theta
    )
    theta_orig = theta.clone()
    U = -theta.clone()
    syre_wd_inplace(
        theta, gamma=0.4, seed1=99, std=0.1,
        seed2=0, d_bound=0.0, advanced_removal=False, offset_base=0,
        U=U,
    )
    torch.testing.assert_close(theta, theta_orig, atol=0, rtol=0)
