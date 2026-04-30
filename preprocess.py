import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from config import PATHS, PHYSICS, TRAIN


class MinMaxScaler:
    """Channel-wise min-max normalization to [0,1]. Fit on training data only."""

    def __init__(self):
        self.min_ = None
        self.max_ = None

    def fit(self, arr):
        self.min_ = arr.min(axis=0, keepdims=True).astype(np.float32)
        self.max_ = arr.max(axis=0, keepdims=True).astype(np.float32)
        return self

    def transform(self, arr):
        return ((arr - self.min_) / (self.max_ - self.min_ + 1e-10)).astype(np.float32)

    def inverse_transform(self, arr):
        return (arr * (self.max_ - self.min_ + 1e-10) + self.min_).astype(np.float32)

    def fit_transform(self, arr):
        return self.fit(arr).transform(arr)


# ---------------------------------------------------------------------------
# Helper: extract the most DYNAMIC chronological segment from a split
# ---------------------------------------------------------------------------

def _make_traj_segment(orig_indices, Wa_all, Wb_all, u_all, pH_all,
                       dt_eff, N=500):
    """
    Given the sorted original time-indices that belong to a split:

    1. Find all consecutive runs (contiguous time-steps with gap == 1).
    2. Among runs long enough to plot, pick the one with the HIGHEST
       pH variance -- this guarantees the trajectory shows real dynamic
       behaviour (acidic transitions, setpoint changes) rather than a
       flat alkaline/acid plateau.
    3. Trim to N+1 points (N inputs + N targets for one-step-ahead).

    Parameters
    ----------
    orig_indices : 1-D int array
        Shuffled original time-indices belonging to this split.
    Wa_all, Wb_all, u_all, pH_all : 1-D float array
        Full subsampled time-series (before any shuffling).
    dt_eff : float
        Effective sampling period (seconds).
    N : int
        Desired trajectory length in steps.

    Returns
    -------
    dict with keys  Wa, Wb, u, pH, t  (all length N+1 or shorter).
    """
    sorted_idx = np.sort(orig_indices)
    diffs      = np.diff(sorted_idx)

    # Locate where consecutive runs break (gap > 1 in time-index)
    break_pts = np.where(diffs != 1)[0] + 1
    runs      = np.split(sorted_idx, break_pts)

    # Only consider runs long enough to make a meaningful plot
    min_len = max(10, N // 2)
    valid   = [r for r in runs if len(r) >= min_len]

    if not valid:
        # Extreme fallback: use the full sorted index
        valid = [sorted_idx]

    # ── Pick the run whose pH segment has the highest variance ────────────
    # High variance means the trajectory spans multiple pH regimes
    # (acid, neutral, alkaline transitions) -- far more informative than
    # a flat plateau.
    def ph_var(run):
        end = min(len(run), N + 1)
        return float(np.var(pH_all[run[:end]]))

    best_run = max(valid, key=ph_var)

    # Trim to at most N+1 points
    seg_idx = best_run[: N + 1]

    print(f"start={seg_idx[0]:>6}  len={len(seg_idx):>4}  "
          f"pH=[{pH_all[seg_idx].min():.2f}, {pH_all[seg_idx].max():.2f}]")

    return {
        "Wa": Wa_all[seg_idx],
        "Wb": Wb_all[seg_idx],
        "u":  u_all[seg_idx],
        "pH": pH_all[seg_idx],
        "t":  np.arange(len(seg_idx)) * dt_eff,
    }


# ---------------------------------------------------------------------------
# Main dataloader builder
# ---------------------------------------------------------------------------

def get_dataloaders(device="cpu"):
    """
    Full pipeline:
        load -> subsample -> one-step pairs -> shuffle (fixed seed)
        -> split (70 / 10 / 20) -> normalise -> DataLoaders
        -> three dynamic trajectory segments (one per split)

    Returns
    -------
    train_loader, val_loader, test_loader,
    scaler_X, scaler_Y, dt_eff,
    test_arrays,
    traj_segment_train, traj_segment_val, traj_segment_test
    """

    # ── Load & subsample ──────────────────────────────────────────────────
    df = pd.read_csv(PATHS["data"])
    print(f"[data] Raw rows   : {len(df):,}  |  cols: {list(df.columns)}")

    step   = PHYSICS["subsample_step"]
    df     = df.iloc[::step].reset_index(drop=True)
    dt_eff = PHYSICS["dt_raw"] * step
    print(f"[data] Subsampled : {len(df):,} rows  |  dt_eff = {dt_eff} s")

    Wa_all = df["Wa"].values.astype(np.float32)
    Wb_all = df["Wb"].values.astype(np.float32)
    u_all  = df["u"].values.astype(np.float32)
    pH_all = df["pH"].values.astype(np.float32)

    # ── One-step-ahead pairs ──────────────────────────────────────────────
    # orig_idx[i] = i  means pair i was built from time-step i in the
    # subsampled array.  We track this through shuffling so we can
    # identify which original time-steps belong to each split.
    X        = np.stack([Wa_all[:-1], Wb_all[:-1], u_all[:-1]], axis=1)
    Y_states = np.stack([Wa_all[1:],  Wb_all[1:]],              axis=1)
    pH_next  = pH_all[1:]
    orig_idx = np.arange(len(X), dtype=np.int64)

    print(f"[data] Pairs      : {len(X):,}  |  "
          f"pH: {pH_next.min():.3f} - {pH_next.max():.3f}")

    # ── Shuffle (fixed seed for full reproducibility) ─────────────────────
    rng  = np.random.default_rng(TRAIN["seed"])
    perm = rng.permutation(len(X))
    X, Y_states, pH_next, orig_idx = (
        X[perm], Y_states[perm], pH_next[perm], orig_idx[perm]
    )

    # ── Split ─────────────────────────────────────────────────────────────
    N       = len(X)
    n_test  = int(N * TRAIN["test_frac"])
    n_val   = int(N * TRAIN["val_frac"])
    n_train = N - n_test - n_val

    sl_tr = slice(0,       n_train)
    sl_va = slice(n_train, n_train + n_val)
    sl_te = slice(n_train + n_val, None)

    X_tr,  Ys_tr,  pH_tr  = X[sl_tr],  Y_states[sl_tr],  pH_next[sl_tr]
    X_va,  Ys_va,  pH_va  = X[sl_va],  Y_states[sl_va],  pH_next[sl_va]
    X_te,  Ys_te,  pH_te  = X[sl_te],  Y_states[sl_te],  pH_next[sl_te]

    idx_tr = orig_idx[sl_tr]
    idx_va = orig_idx[sl_va]
    idx_te = orig_idx[sl_te]

    print(f"[data] Split      : train={n_train:,}  "
          f"val={len(X_va):,}  test={len(X_te):,}")

    # ── Normalisation (fit on training data ONLY) ─────────────────────────
    scaler_X = MinMaxScaler().fit(X_tr)
    scaler_Y = MinMaxScaler().fit(Ys_tr)

    # pH bounds computed over full dataset so inverse-transform is
    # consistent regardless of which split is being evaluated.
    scaler_X.pH_min = float(pH_next.min())
    scaler_X.pH_max = float(pH_next.max())

    def norm_pH(x):
        return ((x - scaler_X.pH_min) /
                (scaler_X.pH_max - scaler_X.pH_min + 1e-10)).astype(np.float32)

    def make_tensors(Xa, Ysa, pHa):
        X_n   = torch.tensor(scaler_X.transform(Xa),  device=device)
        Ys_n  = torch.tensor(scaler_Y.transform(Ysa), device=device)
        pH_n  = torch.tensor(norm_pH(pHa),             device=device)
        Y_all = torch.cat([Ys_n, pH_n.unsqueeze(1)], dim=1)
        return X_n, Y_all

    X_tr_t, Y_tr_t = make_tensors(X_tr, Ys_tr, pH_tr)
    X_va_t, Y_va_t = make_tensors(X_va, Ys_va, pH_va)
    X_te_t, Y_te_t = make_tensors(X_te, Ys_te, pH_te)

    bs = TRAIN["batch_size"]
    train_loader = DataLoader(TensorDataset(X_tr_t, Y_tr_t),
                              batch_size=bs,   shuffle=True,  drop_last=True)
    val_loader   = DataLoader(TensorDataset(X_va_t, Y_va_t),
                              batch_size=bs*2, shuffle=False)
    test_loader  = DataLoader(TensorDataset(X_te_t, Y_te_t),
                              batch_size=bs*2, shuffle=False)

    test_arrays = {
        "X":        X_te,
        "Y_states": Ys_te,
        "pH":       pH_te,
    }

    # ── Trajectory segments ───────────────────────────────────────────────
    # Each segment is drawn from its own split's time-indices and is
    # chosen to maximise pH variance (most dynamic / interesting region).
    print("[data] Selecting trajectory segments (highest-variance runs):")
    print("[data]   TRAIN  ", end="")
    traj_train = _make_traj_segment(idx_tr, Wa_all, Wb_all, u_all, pH_all,
                                    dt_eff, N=500)
    print("[data]   VAL    ", end="")
    traj_val   = _make_traj_segment(idx_va, Wa_all, Wb_all, u_all, pH_all,
                                    dt_eff, N=500)
    print("[data]   TEST   ", end="")
    traj_test  = _make_traj_segment(idx_te, Wa_all, Wb_all, u_all, pH_all,
                                    dt_eff, N=500)

    return (train_loader, val_loader, test_loader,
            scaler_X, scaler_Y, dt_eff,
            test_arrays,
            traj_train, traj_val, traj_test)