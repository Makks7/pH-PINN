import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import time

from config     import TRAIN, PATHS, PHYSICS, MODEL
from model      import PINN, model_summary
from preprocess import get_dataloaders
from physics    import physics_loss_torch

torch.manual_seed(TRAIN["seed"])
np.random.seed(TRAIN["seed"])

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[train] Device: {device}")

Path(PATHS["model_ckpt"]).parent.mkdir(parents=True, exist_ok=True)
Path(PATHS["results"]).mkdir(parents=True, exist_ok=True)

PLT = {
    "true_color":  "#1f77b4",
    "pred_color":  "#d62728",
    "ctrl_color":  "#2ca02c",
    "val_color":   "#ff7f0e",
    "font_title":  13,
    "font_label":  11,
    "font_tick":   10,
    "font_legend": 10,
    "lw_true":     1.8,
    "lw_pred":     1.5,
    "dpi":         200,
}


# ── Loss ──────────────────────────────────────────────────────────────────────

def compute_loss(X_b, Y_b, Y_pred, scaler_X, scaler_Y,
                 dt, w_state, w_phys, w_pH, mse_fn):
    l_state = mse_fn(Y_pred[:, 0], Y_b[:, 0]) + mse_fn(Y_pred[:, 1], Y_b[:, 1])
    l_pH    = mse_fn(Y_pred[:, 2], Y_b[:, 2])
    l_phys  = physics_loss_torch(
        X_b[:, 0], X_b[:, 1], X_b[:, 2],
        Y_pred[:, 0], Y_pred[:, 1],
        scaler_X, scaler_Y, dt,
    )
    total = w_state * l_state + w_pH * l_pH + w_phys * l_phys
    return total, l_state.item(), l_pH.item(), l_phys.item()


def run_epoch(model, loader, optimizer, scaler_X, scaler_Y,
              dt, w_state, w_phys, w_pH, training=True):
    model.train(training)
    mse_fn = nn.MSELoss()
    total, n = 0.0, 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for X_b, Y_b in loader:
            Y_pred = model(X_b)
            loss, _, _, _ = compute_loss(
                X_b, Y_b, Y_pred, scaler_X, scaler_Y,
                dt, w_state, w_phys, w_pH, mse_fn
            )
            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total += loss.item()
            n     += 1
    return total / max(n, 1)


# ── Metrics ───────────────────────────────────────────────────────────────────

def get_metrics(model, loader, scaler_X, scaler_Y):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for X_b, Y_b in loader:
            preds.append(model(X_b).cpu().numpy())
            trues.append(Y_b.cpu().numpy())

    P = np.concatenate(preds, axis=0)
    T = np.concatenate(trues, axis=0)

    pH_range = scaler_X.pH_max - scaler_X.pH_min + 1e-10

    Wa_p = scaler_Y.inverse_transform(P[:, :2])[:, 0]
    Wb_p = scaler_Y.inverse_transform(P[:, :2])[:, 1]
    pH_p = P[:, 2] * pH_range + scaler_X.pH_min

    Wa_t = scaler_Y.inverse_transform(T[:, :2])[:, 0]
    Wb_t = scaler_Y.inverse_transform(T[:, :2])[:, 1]
    pH_t = T[:, 2] * pH_range + scaler_X.pH_min

    def rmse(a, b): return float(np.sqrt(np.mean((a - b)**2)))
    def mae(a, b):  return float(np.mean(np.abs(a - b)))
    def r2(a, b):
        return float(1 - np.sum((b-a)**2) / (np.sum((b-np.mean(b))**2) + 1e-12))

    return {
        "Wa": {"rmse": rmse(Wa_p,Wa_t), "mae": mae(Wa_p,Wa_t), "r2": r2(Wa_p,Wa_t),
               "pred": Wa_p, "true": Wa_t},
        "Wb": {"rmse": rmse(Wb_p,Wb_t), "mae": mae(Wb_p,Wb_t), "r2": r2(Wb_p,Wb_t),
               "pred": Wb_p, "true": Wb_t},
        "pH": {"rmse": rmse(pH_p,pH_t), "mae": mae(pH_p,pH_t), "r2": r2(pH_p,pH_t),
               "pred": pH_p, "true": pH_t},
    }


def print_metrics_table(label, m):
    print(f"\n  [{label}]")
    print(f"  {'':10s} {'Wa':>12}  {'Wb':>12}  {'pH':>10}")
    print(f"  {'-'*50}")
    print(f"  {'RMSE':<10} {m['Wa']['rmse']:>12.6f}  "
          f"{m['Wb']['rmse']:>12.6f}  {m['pH']['rmse']:>10.4f}")
    print(f"  {'MAE':<10} {m['Wa']['mae']:>12.6f}  "
          f"{m['Wb']['mae']:>12.6f}  {m['pH']['mae']:>10.4f}")
    print(f"  {'R2':<10} {m['Wa']['r2']:>12.4f}  "
          f"{m['Wb']['r2']:>12.4f}  {m['pH']['r2']:>10.4f}")


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_training_curve(history, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    epochs = range(1, len(history["train"]) + 1)

    axes[0].plot(epochs, history["train"], color=PLT["true_color"],
                 lw=1.5, label="Training loss")
    axes[0].plot(epochs, history["val"], color=PLT["val_color"],
                 lw=1.5, linestyle="--", label="Validation loss")
    axes[0].set_xlabel("Epoch", fontsize=PLT["font_label"])
    axes[0].set_ylabel("Loss", fontsize=PLT["font_label"])
    axes[0].set_title("Training & Validation Loss (Linear)", fontsize=PLT["font_title"])
    axes[0].legend(fontsize=PLT["font_legend"])
    axes[0].tick_params(labelsize=PLT["font_tick"])
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(epochs, history["train"], color=PLT["true_color"],
                     lw=1.5, label="Training loss")
    axes[1].semilogy(epochs, history["val"], color=PLT["val_color"],
                     lw=1.5, linestyle="--", label="Validation loss")
    axes[1].set_xlabel("Epoch", fontsize=PLT["font_label"])
    axes[1].set_ylabel("Loss (log scale)", fontsize=PLT["font_label"])
    axes[1].set_title("Training & Validation Loss (Log Scale)", fontsize=PLT["font_title"])
    axes[1].legend(fontsize=PLT["font_legend"])
    axes[1].tick_params(labelsize=PLT["font_tick"])
    axes[1].grid(True, alpha=0.3, which="both")

    plt.suptitle("PINN Training Convergence -- Hensen-Seborg pH Model",
                 fontsize=PLT["font_title"] + 1, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=PLT["dpi"], bbox_inches="tight")
    plt.close()
    print(f"[train] Training curve  -> {save_path}")


def plot_trajectory(model, scaler_X, scaler_Y, seg, tag, save_dir):
    """
    Open-loop one-step-ahead trajectory on the most dynamic segment
    from the correct split.  Four panels: pH, Wa, Wb, control u.
    """
    model.eval()
    Wa = seg["Wa"]; Wb = seg["Wb"]
    u  = seg["u"];  pH = seg["pH"]
    t  = seg["t"]

    N   = len(Wa) - 1
    X   = np.stack([Wa[:N], Wb[:N], u[:N]], axis=1).astype(np.float32)
    X_n = torch.tensor(scaler_X.transform(X), device=device)

    with torch.no_grad():
        Y_n = model(X_n).cpu().numpy()

    pH_range = scaler_X.pH_max - scaler_X.pH_min + 1e-10
    Wa_pred  = scaler_Y.inverse_transform(Y_n[:, :2])[:, 0]
    Wb_pred  = scaler_Y.inverse_transform(Y_n[:, :2])[:, 1]
    pH_pred  = Y_n[:, 2] * pH_range + scaler_X.pH_min

    Wa_true = Wa[1:N+1]; Wb_true = Wb[1:N+1]; pH_true = pH[1:N+1]
    t_plot  = t[1:N+1]

    def rmse(a, b): return float(np.sqrt(np.mean((a-b)**2)))
    def mae(a, b):  return float(np.mean(np.abs(a-b)))
    def r2(a, b):
        return float(1 - np.sum((b-a)**2) / (np.sum((b-np.mean(b))**2) + 1e-12))

    fig = plt.figure(figsize=(13, 12))
    gs  = gridspec.GridSpec(4, 1, hspace=0.45)

    ax0 = fig.add_subplot(gs[0])
    ax0.plot(t_plot, pH_true, color=PLT["true_color"], lw=PLT["lw_true"],
             label="True (Simulink)", zorder=3)
    ax0.plot(t_plot, pH_pred, color=PLT["pred_color"], lw=PLT["lw_pred"],
             linestyle="--", label="PINN predicted", zorder=4)
    ax0.set_ylabel("pH", fontsize=PLT["font_label"])
    ax0.set_title(
        f"pH  |  R\u00b2 = {r2(pH_pred,pH_true):.4f}  |  "
        f"RMSE = {rmse(pH_pred,pH_true):.4f}  |  MAE = {mae(pH_pred,pH_true):.4f}",
        fontsize=PLT["font_title"])
    ax0.legend(fontsize=PLT["font_legend"], loc="upper right")
    ax0.tick_params(labelsize=PLT["font_tick"])
    ax0.grid(True, alpha=0.3)

    ax1 = fig.add_subplot(gs[1])
    ax1.plot(t_plot, Wa_true, color=PLT["true_color"], lw=PLT["lw_true"],
             label="True Wa", zorder=3)
    ax1.plot(t_plot, Wa_pred, color=PLT["pred_color"], lw=PLT["lw_pred"],
             linestyle="--", label="PINN predicted", zorder=4)
    ax1.set_ylabel("$W_a$ (mol)", fontsize=PLT["font_label"])
    ax1.set_title(
        f"$W_a$  |  R\u00b2 = {r2(Wa_pred,Wa_true):.4f}  |  "
        f"RMSE = {rmse(Wa_pred,Wa_true):.6f}  |  MAE = {mae(Wa_pred,Wa_true):.6f}",
        fontsize=PLT["font_title"])
    ax1.legend(fontsize=PLT["font_legend"], loc="upper right")
    ax1.tick_params(labelsize=PLT["font_tick"])
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[2])
    ax2.plot(t_plot, Wb_true, color=PLT["true_color"], lw=PLT["lw_true"],
             label="True Wb", zorder=3)
    ax2.plot(t_plot, Wb_pred, color=PLT["pred_color"], lw=PLT["lw_pred"],
             linestyle="--", label="PINN predicted", zorder=4)
    ax2.set_ylabel("$W_b$ (mol)", fontsize=PLT["font_label"])
    ax2.set_title(
        f"$W_b$  |  R\u00b2 = {r2(Wb_pred,Wb_true):.4f}  |  "
        f"RMSE = {rmse(Wb_pred,Wb_true):.6f}  |  MAE = {mae(Wb_pred,Wb_true):.6f}",
        fontsize=PLT["font_title"])
    ax2.legend(fontsize=PLT["font_legend"], loc="upper right")
    ax2.tick_params(labelsize=PLT["font_tick"])
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[3])
    ax3.step(t_plot, u[:N], color=PLT["ctrl_color"], lw=1.4,
             where="post", label="Base flow rate $u_1$ (ml/s)")
    ax3.set_ylabel("$u_1$ (ml/s)", fontsize=PLT["font_label"])
    ax3.set_xlabel("Time (s)", fontsize=PLT["font_label"])
    ax3.set_title("Manipulated Input -- Base Flow Rate", fontsize=PLT["font_title"])
    ax3.legend(fontsize=PLT["font_legend"], loc="upper right")
    ax3.tick_params(labelsize=PLT["font_tick"])
    ax3.grid(True, alpha=0.3)

    fig.suptitle(
        f"PINN Open-Loop One-Step-Ahead Prediction -- "
        f"Hensen-Seborg  [{tag.upper()} segment]",
        fontsize=PLT["font_title"] + 2, fontweight="bold", y=1.01
    )

    path = os.path.join(save_dir, f"trajectory_{tag}.png")
    plt.savefig(path, dpi=PLT["dpi"], bbox_inches="tight")
    plt.close()
    print(f"[train] Trajectory ({tag}) -> {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def train():
    print("\n" + "="*60)
    print("  PINN  --  Hensen-Seborg pH Neutralization")
    print("="*60 + "\n")

    (train_loader, val_loader, test_loader,
     scaler_X, scaler_Y, dt_eff,
     test_arrays,
     traj_train, traj_val, traj_test) = get_dataloaders(device=device)

    print(f"\n[train] Effective dt : {dt_eff} s")

    model = PINN().to(device)
    model_summary(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=TRAIN["lr_patience"], factor=TRAIN["lr_factor"]
    )

    w_state = TRAIN["w_state"]
    w_phys  = TRAIN["w_phys"]
    w_pH    = TRAIN["w_pH"]

    history    = {"train": [], "val": []}
    best_val   = float("inf")
    no_improve = 0

    print(f"\n{'Ep':>6} | {'Train':>10} | {'Val':>10} | {'w_phys':>8} | {'LR':>9}")
    print("-" * 54)

    t0 = time.time()

    for ep in range(1, TRAIN["epochs"] + 1):

        ramp       = min(1.0, ep / max(1, 0.20 * TRAIN["epochs"]))
        w_phys_eff = w_phys * ramp

        tr = run_epoch(model, train_loader, optimizer, scaler_X, scaler_Y,
                       dt_eff, w_state, w_phys_eff, w_pH, training=True)
        va = run_epoch(model, val_loader,   optimizer, scaler_X, scaler_Y,
                       dt_eff, w_state, w_phys_eff, w_pH, training=False)

        history["train"].append(tr)
        history["val"].append(va)
        scheduler.step(va)
        lr = optimizer.param_groups[0]["lr"]

        if ep % 50 == 0 or ep == 1:
            print(f"{ep:>6} | {tr:>10.5f} | {va:>10.5f} | "
                  f"{w_phys_eff:>8.4f} | {lr:>9.2e}")

        if va < best_val:
            best_val   = va
            no_improve = 0
            torch.save({
                "epoch":        ep,
                "model_state":  model.state_dict(),
                "val_loss":     best_val,
                "scaler_X_min": scaler_X.min_,
                "scaler_X_max": scaler_X.max_,
                "scaler_Y_min": scaler_Y.min_,
                "scaler_Y_max": scaler_Y.max_,
                "pH_min":       scaler_X.pH_min,
                "pH_max":       scaler_X.pH_max,
                "dt_eff":       dt_eff,
                "model_config": MODEL,
            }, PATHS["model_ckpt"])
        else:
            no_improve += 1

        if no_improve >= TRAIN["early_stop"]:
            print(f"\n[train] Early stop at epoch {ep}  "
                  f"(no gain for {TRAIN['early_stop']} epochs)")
            break

    elapsed = time.time() - t0
    print(f"\n[train] Best val loss : {best_val:.6f}")
    print(f"[train] Time          : {elapsed:.1f} s  ({elapsed/60:.1f} min)")
    print(f"[train] Checkpoint    : {PATHS['model_ckpt']}")

    # Reload best checkpoint for final metrics and plots
    ckpt = torch.load(PATHS["model_ckpt"], map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    print("\n" + "="*54)
    print("  FINAL METRICS -- Best Checkpoint")
    print("="*54)
    print_metrics_table("Training set",   get_metrics(model, train_loader, scaler_X, scaler_Y))
    print_metrics_table("Validation set", get_metrics(model, val_loader,   scaler_X, scaler_Y))
    print()

    plot_training_curve(history, os.path.join(PATHS["results"], "training_curve.png"))

    # Each plot uses its own split's most dynamic segment
    plot_trajectory(model, scaler_X, scaler_Y, traj_train, "train", PATHS["results"])


if __name__ == "__main__":
    train()