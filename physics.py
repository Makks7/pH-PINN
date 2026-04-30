import numpy as np
import torch
import torch.nn as nn
from config import PHYSICS

V   = PHYSICS["V"]
u3  = PHYSICS["u3"];  u2  = PHYSICS["u2"]
Wa3 = PHYSICS["Wa3"]; Wa1 = PHYSICS["Wa1"]; Wa2 = PHYSICS["Wa2"]
Wb3 = PHYSICS["Wb3"]; Wb1 = PHYSICS["Wb1"]; Wb2 = PHYSICS["Wb2"]
pk1 = PHYSICS["pk1"]; pk2 = PHYSICS["pk2"]


def physics_loss_torch(Wa_k_n, Wb_k_n, u1_k_n,
                       Wa_pred_n, Wb_pred_n,
                       scaler_X, scaler_Y, dt):
    """
    Proper differentiable ODE residual loss -- gradients flow through network.

    Inverse-transform inputs/outputs to physical units using torch ops
    so the computation stays on the autograd graph.

    Hensen-Seborg ODEs:
      dWa/dt = (1/V)*[u3*(Wa3-Wa) + u1*(Wa1-Wa) + u2*(Wa2-Wa)]
      dWb/dt = (1/V)*[u3*(Wb3-Wb) + u1*(Wb1-Wb) + u2*(Wb2-Wb)]

    Residual (should be zero if NN is physically consistent):
      R_Wa = (Wa_pred - Wa_k)/dt  -  dWa/dt_physics
      R_Wb = (Wb_pred - Wb_k)/dt  -  dWb/dt_physics
    """
    dev = Wa_k_n.device

    # Scaler params as tensors (keeps computation on graph)
    Xmin = torch.tensor(scaler_X.min_, dtype=torch.float32, device=dev)  # (1,3)
    Xmax = torch.tensor(scaler_X.max_, dtype=torch.float32, device=dev)
    Ymin = torch.tensor(scaler_Y.min_, dtype=torch.float32, device=dev)  # (1,2)
    Ymax = torch.tensor(scaler_Y.max_, dtype=torch.float32, device=dev)

    Xrange = Xmax - Xmin + 1e-10
    Yrange = Ymax - Ymin + 1e-10

    # Inverse-transform inputs to physical units (differentiable)
    X_n = torch.stack([Wa_k_n, Wb_k_n, u1_k_n], dim=1)        # (B,3)
    X_p = X_n * Xrange + Xmin

    Wa_k = X_p[:, 0]; Wb_k = X_p[:, 1]; u1_k = X_p[:, 2]

    # Inverse-transform predictions to physical units (differentiable)
    Y_n = torch.stack([Wa_pred_n, Wb_pred_n], dim=1)            # (B,2)
    Y_p = Y_n * Yrange + Ymin

    Wa_p = Y_p[:, 0]; Wb_p = Y_p[:, 1]

    # Physics RHS at current state
    dWa_phys = (1.0/V) * (u3*(Wa3-Wa_k) + u1_k*(Wa1-Wa_k) + u2*(Wa2-Wa_k))
    dWb_phys = (1.0/V) * (u3*(Wb3-Wb_k) + u1_k*(Wb1-Wb_k) + u2*(Wb2-Wb_k))

    # Numerical derivatives from NN predictions
    dWa_nn = (Wa_p - Wa_k) / dt
    dWb_nn = (Wb_p - Wb_k) / dt

    R_Wa = dWa_nn - dWa_phys
    R_Wb = dWb_nn - dWb_phys

    # Normalize by detached RMS to keep loss O(1) without breaking gradients
    s_Wa = torch.sqrt(torch.mean(dWa_phys.detach()**2)) + 1e-15
    s_Wb = torch.sqrt(torch.mean(dWb_phys.detach()**2)) + 1e-15

    return torch.mean((R_Wa/s_Wa)**2) + torch.mean((R_Wb/s_Wb)**2)


def _pH_residual(pH, Wa, Wb):
    """Eq. 26 residual from Hensen-Seborg / supervisor's paper."""
    num = 1.0 + 2.0 * 10.0**(pH - pk2)
    den = 1.0 + 10.0**(pk1 - pH) + 10.0**(pH - pk2)
    return Wa + 10.0**(pH - 14.0) - 10.0**(-pH) + Wb * (num / den)


def solve_pH_numpy(Wa_arr, Wb_arr, tol=1e-8, max_iter=80):
    """Bisection solver for pH from (Wa, Wb). Used in test.py cross-verification only."""
    pH_out = np.empty(len(Wa_arr), dtype=np.float64)
    for i in range(len(Wa_arr)):
        lo, hi = 1.0, 14.0
        for _ in range(max_iter):
            mid   = 0.5 * (lo + hi)
            f_lo  = _pH_residual(lo,  Wa_arr[i], Wb_arr[i])
            f_mid = _pH_residual(mid, Wa_arr[i], Wb_arr[i])
            if f_lo * f_mid <= 0.0:
                hi = mid
            else:
                lo = mid
            if (hi - lo) < tol:
                break
        pH_out[i] = 0.5 * (lo + hi)
    return pH_out