import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config     import PATHS
from model      import PINN
from preprocess import get_dataloaders

device = "cuda" if torch.cuda.is_available() else "cpu"

PLT = {
    "true_color": "#1f77b4",
    "pred_color": "#d62728",
    "ctrl_color": "#2ca02c",
    "font_title": 13, "font_label": 11,
    "font_tick":  10, "font_legend": 10,
    "lw_true": 1.8,  "lw_pred": 1.5,
    "dpi": 200,
}


def _denorm(Y_n, scaler_X, scaler_Y):
    pH_range = scaler_X.pH_max - scaler_X.pH_min + 1e-10
    Wa = scaler_Y.inverse_transform(Y_n[:, :2])[:, 0]
    Wb = scaler_Y.inverse_transform(Y_n[:, :2])[:, 1]
    pH = Y_n[:, 2] * pH_range + scaler_X.pH_min
    return Wa, Wb, pH


def _metrics(pred, true):
    rmse = float(np.sqrt(np.mean((pred - true)**2)))
    mae  = float(np.mean(np.abs(pred - true)))
    r2   = float(1 - np.sum((true-pred)**2) / (np.sum((true-np.mean(true))**2) + 1e-12))
    return rmse, mae, r2


def plot_parity(results, save_path):
    """Publication-quality three-panel parity plot for the test set."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, key, label, unit in zip(
        axes,
        ["pH",  "Wa",     "Wb"],
        ["pH",  "$W_a$",  "$W_b$"],
        ["",    " (mol)", " (mol)"]
    ):
        true = results[key]["true"]
        pred = results[key]["pred"]
        rmse, mae, r2 = results[key]["rmse"], results[key]["mae"], results[key]["r2"]

        ax.scatter(true, pred, s=6, alpha=0.4, color=PLT["true_color"],
                   rasterized=True, label="Predictions")
        lims = [min(true.min(), pred.min()) * 0.98,
                max(true.max(), pred.max()) * 1.02]
        ax.plot(lims, lims, color=PLT["pred_color"], lw=1.8,
                linestyle="--", label="Perfect fit")
        ax.set_xlabel(f"True {label}{unit}", fontsize=PLT["font_label"])
        ax.set_ylabel(f"Predicted {label}{unit}", fontsize=PLT["font_label"])
        ax.set_title(
            f"{label}  |  R\u00b2 = {r2:.4f}  |  "
            f"RMSE = {rmse:.5f}  |  MAE = {mae:.5f}",
            fontsize=PLT["font_title"] - 1
        )
        ax.legend(fontsize=PLT["font_legend"])
        ax.tick_params(labelsize=PLT["font_tick"])
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Parity Plots -- PINN (Hensen-Seborg)  [Test Set]",
        fontsize=PLT["font_title"] + 1, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=PLT["dpi"], bbox_inches="tight")
    plt.close()
    print(f"[test] Parity plot    -> {save_path}")


def plot_residuals(results, save_path):
    """Residual histograms for all three outputs."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, key, label in zip(
        axes,
        ["pH",  "Wa",          "Wb"],
        ["pH",  "$W_a$ (mol)", "$W_b$ (mol)"]
    ):
        res  = results[key]["pred"] - results[key]["true"]
        rmse = results[key]["rmse"]
        mae  = results[key]["mae"]

        ax.hist(res, bins=60, color=PLT["true_color"],
                edgecolor="white", linewidth=0.3, density=True)
        ax.axvline(0, color=PLT["pred_color"], linestyle="--", lw=1.8, label="Zero error")
        ax.axvline(np.mean(res), color="#ff7f0e", linestyle=":", lw=1.5,
                   label=f"Mean = {np.mean(res):.5f}")
        ax.set_xlabel(f"Residual -- {label}", fontsize=PLT["font_label"])
        ax.set_ylabel("Density", fontsize=PLT["font_label"])
        ax.set_title(
            f"{label}  |  RMSE = {rmse:.5f}  |  MAE = {mae:.5f}",
            fontsize=PLT["font_title"]
        )
        ax.legend(fontsize=PLT["font_legend"])
        ax.tick_params(labelsize=PLT["font_tick"])
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Prediction Residuals -- PINN (Hensen-Seborg)  [Test Set]",
        fontsize=PLT["font_title"] + 1, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=PLT["dpi"], bbox_inches="tight")
    plt.close()
    print(f"[test] Residuals      -> {save_path}")


def plot_trajectory(model, scaler_X, scaler_Y, seg, save_path):
    """
    Open-loop one-step-ahead trajectory on the TEST split segment.
    This segment is genuinely from held-out test time-steps never
    seen during training or validation.
    """
    model.eval()
    Wa = seg["Wa"]; Wb = seg["Wb"]
    u  = seg["u"];  pH = seg["pH"]
    t  = seg["t"]
    N  = len(Wa) - 1

    X   = np.stack([Wa[:N], Wb[:N], u[:N]], axis=1).astype(np.float32)
    X_n = torch.tensor(scaler_X.transform(X), device=device)

    with torch.no_grad():
        Y_n = model(X_n).cpu().numpy()

    Wa_pred, Wb_pred, pH_pred = _denorm(Y_n, scaler_X, scaler_Y)
    Wa_true = Wa[1:N+1]; Wb_true = Wb[1:N+1]; pH_true = pH[1:N+1]
    t_plot  = t[1:N+1]

    def r2(p, t):
        return float(1 - np.sum((t-p)**2) / (np.sum((t-np.mean(t))**2) + 1e-12))
    def rmse(p, t): return float(np.sqrt(np.mean((p-t)**2)))
    def mae(p, t):  return float(np.mean(np.abs(p-t)))

    fig, axes = plt.subplots(4, 1, figsize=(13, 12))
    plt.subplots_adjust(hspace=0.45)

    for ax, true, pred, ylabel in zip(
        axes[:3],
        [pH_true,  Wa_true,       Wb_true],
        [pH_pred,  Wa_pred,       Wb_pred],
        ["pH",     "$W_a$ (mol)", "$W_b$ (mol)"]
    ):
        ax.plot(t_plot, true, color=PLT["true_color"], lw=PLT["lw_true"],
                label="True (Simulink)", zorder=3)
        ax.plot(t_plot, pred, color=PLT["pred_color"], lw=PLT["lw_pred"],
                linestyle="--", label="PINN predicted", zorder=4)
        ax.set_ylabel(ylabel, fontsize=PLT["font_label"])
        ax.set_title(
            f"{ylabel.split('(')[0].strip()}  |  "
            f"R\u00b2 = {r2(pred,true):.4f}  |  "
            f"RMSE = {rmse(pred,true):.5f}  |  "
            f"MAE = {mae(pred,true):.5f}",
            fontsize=PLT["font_title"]
        )
        ax.legend(fontsize=PLT["font_legend"], loc="upper right")
        ax.tick_params(labelsize=PLT["font_tick"])
        ax.grid(True, alpha=0.3)

    axes[3].step(t_plot, u[:N], color=PLT["ctrl_color"], lw=1.4,
                 where="post", label="Base flow rate $u_1$ (ml/s)")
    axes[3].set_ylabel("$u_1$ (ml/s)", fontsize=PLT["font_label"])
    axes[3].set_xlabel("Time (s)", fontsize=PLT["font_label"])
    axes[3].set_title("Manipulated Input -- Base Flow Rate", fontsize=PLT["font_title"])
    axes[3].legend(fontsize=PLT["font_legend"], loc="upper right")
    axes[3].tick_params(labelsize=PLT["font_tick"])
    axes[3].grid(True, alpha=0.3)

    fig.suptitle(
        "PINN One-Step-Ahead Open-Loop Response -- Hensen-Seborg  [TEST Segment]",
        fontsize=PLT["font_title"] + 2, fontweight="bold", y=1.01
    )
    plt.savefig(save_path, dpi=PLT["dpi"], bbox_inches="tight")
    plt.close()
    print(f"[test] Trajectory     -> {save_path}")


# ── Multi-step rollout ──────────────────────────────────────────────────────

def plot_multistep_rollout(model, scaler_X, scaler_Y, seg, save_path,
                           horizon=50):
    """
    Closed-loop multi-step rollout: PINN predictions fed back as inputs.
    Tests error accumulation -- critical for NMPC viability.

    horizon : number of steps to roll out autoregressively.
    """
    model.eval()
    Wa = seg["Wa"]; Wb = seg["Wb"]
    u  = seg["u"];  pH = seg["pH"]
    t  = seg["t"]

    horizon = min(horizon, len(Wa) - 1)

    pH_range = scaler_X.pH_max - scaler_X.pH_min + 1e-10

    # True trajectories (reference)
    pH_true = pH[1: horizon + 1]
    Wa_true = Wa[1: horizon + 1]
    Wb_true = Wb[1: horizon + 1]
    t_plot  = t[1: horizon + 1]

    # Initialise rollout from the first true state
    Wa_k = float(Wa[0])
    Wb_k = float(Wb[0])

    pH_roll = []
    Wa_roll = []
    Wb_roll = []

    with torch.no_grad():
        for k in range(horizon):
            x_raw = np.array([[Wa_k, Wb_k, float(u[k])]], dtype=np.float32)
            x_n   = torch.tensor(scaler_X.transform(x_raw), device=device)
            y_n   = model(x_n).cpu().numpy()[0]            # (3,)

            Wa_next = float(scaler_Y.inverse_transform(y_n[:2].reshape(1, 2))[0, 0])
            Wb_next = float(scaler_Y.inverse_transform(y_n[:2].reshape(1, 2))[0, 1])
            pH_next = float(y_n[2] * pH_range + scaler_X.pH_min)

            Wa_roll.append(Wa_next)
            Wb_roll.append(Wb_next)
            pH_roll.append(pH_next)

            # Feed prediction back as next state
            Wa_k = Wa_next
            Wb_k = Wb_next

    Wa_roll = np.array(Wa_roll)
    Wb_roll = np.array(Wb_roll)
    pH_roll = np.array(pH_roll)

    def rmse(a, b): return float(np.sqrt(np.mean((a-b)**2)))
    def mae(a, b):  return float(np.mean(np.abs(a-b)))
    def r2(a, b):
        return float(1 - np.sum((b-a)**2) / (np.sum((b-np.mean(b))**2) + 1e-12))

    fig, axes = plt.subplots(3, 1, figsize=(13, 10))
    plt.subplots_adjust(hspace=0.45)

    for ax, true, pred, ylabel in zip(
        axes,
        [pH_true,  Wa_true,       Wb_true],
        [pH_roll,  Wa_roll,       Wb_roll],
        ["pH",     "$W_a$ (mol)", "$W_b$ (mol)"]
    ):
        ax.plot(t_plot, true, color="#1f77b4", lw=1.8, label="True (Simulink)", zorder=3)
        ax.plot(t_plot, pred, color="#d62728", lw=1.5, linestyle="--",
                label="PINN rollout", zorder=4)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(
            f"{ylabel.split('(')[0].strip()}  |  "
            f"R\u00b2 = {r2(pred,true):.4f}  |  "
            f"RMSE = {rmse(pred,true):.5f}  |  "
            f"MAE = {mae(pred,true):.5f}",
            fontsize=13
        )
        ax.legend(fontsize=10, loc="upper right")
        ax.tick_params(labelsize=10)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)", fontsize=11)

    fig.suptitle(
        f"PINN {horizon}-Step Closed-Loop Rollout -- Hensen-Seborg  [TEST Segment]\n"
        "(Predictions fed back as inputs -- NMPC viability test)",
        fontsize=13, fontweight="bold", y=1.01
    )
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[test] Multi-step rollout ({horizon} steps) -> {save_path}")

    # Summary
    print(f"\n  [Multi-step rollout  |  horizon = {horizon} steps]")
    print(f"  {'':10s} {'pH':>10}  {'Wa':>14}  {'Wb':>14}")
    print(f"  {'-'*52}")
    print(f"  {'RMSE':<10} {rmse(pH_roll,pH_true):>10.4f}  "
          f"{rmse(Wa_roll,Wa_true):>14.6f}  {rmse(Wb_roll,Wb_true):>14.6f}")
    print(f"  {'MAE':<10} {mae(pH_roll,pH_true):>10.4f}  "
          f"{mae(Wa_roll,Wa_true):>14.6f}  {mae(Wb_roll,Wb_true):>14.6f}")
    print(f"  {'R2':<10} {r2(pH_roll,pH_true):>10.4f}  "
          f"{r2(Wa_roll,Wa_true):>14.4f}  {r2(Wb_roll,Wb_true):>14.4f}")


# ── Entry point ───────────────────────────────────────────────────────────────

def run_test():
    print("\n[test] Loading checkpoint ...")
    ckpt = torch.load(PATHS["model_ckpt"], map_location=device, weights_only=False)

    # ── Updated unpacking ──
    (_, _, test_loader, scaler_X, scaler_Y, dt_eff,
     test_arrays, _traj_train, _traj_val, traj_test) = get_dataloaders(device=device)

    model = PINN().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"[test] Epoch : {ckpt['epoch']}  |  val_loss : {ckpt['val_loss']:.6f}")

    preds, trues = [], []
    with torch.no_grad():
        for X_b, Y_b in test_loader:
            preds.append(model(X_b).cpu().numpy())
            trues.append(Y_b.cpu().numpy())

    P = np.concatenate(preds, axis=0)
    T = np.concatenate(trues, axis=0)

    Wa_p, Wb_p, pH_p = _denorm(P, scaler_X, scaler_Y)
    Wa_t, Wb_t, pH_t = _denorm(T, scaler_X, scaler_Y)

    results = {}
    for key, pred, true in [("pH", pH_p, pH_t), ("Wa", Wa_p, Wa_t), ("Wb", Wb_p, Wb_t)]:
        rmse, mae, r2 = _metrics(pred, true)
        results[key] = {"rmse": rmse, "mae": mae, "r2": r2, "pred": pred, "true": true}

    print("\n" + "="*60)
    print("  TEST SET RESULTS -- PINN (Hensen-Seborg)")
    print("="*60)
    print(f"\n  {'Metric':<10} {'Wa':>14}  {'Wb':>14}  {'pH':>12}")
    print(f"  {'-'*56}")
    for m in ["rmse", "mae", "r2"]:
        print(f"  {m.upper():<10} {results['Wa'][m]:>14.6f}  "
              f"{results['Wb'][m]:>14.6f}  {results['pH'][m]:>12.4f}")
    print(f"  {'-'*56}")
    print(f"\n  Test samples : {len(pH_t):,}  |  Best epoch : {ckpt['epoch']}")

    # ── Figures ──
    plot_parity(results,
                os.path.join(PATHS["results"], "test_parity.png"))

    plot_residuals(results,
                   os.path.join(PATHS["results"], "test_residuals.png"))

    # ── Trajectory: genuinely from test split ──
    plot_trajectory(model, scaler_X, scaler_Y, traj_test,
                    os.path.join(PATHS["results"], "test_trajectory.png"))

    # ── Multi-step rollout (NMPC viability test) ──
    plot_multistep_rollout(
        model, scaler_X, scaler_Y, traj_test,
        os.path.join(PATHS["results"], "test_multistep_rollout.png"),
        horizon=50,
    )


if __name__ == "__main__":
    run_test()