# pH-PINN: Physics-Informed Neural Networks for pH Neutralization Control

A PINN-based surrogate model of the Hensen-Seborg pH neutralization CSTR, trained to predict acid/base state dynamics and serve as the prediction model inside a Nonlinear Model Predictive Controller (NMPC) implemented in MATLAB/Simulink.

## Overview

pH control in a continuous stirred-tank reactor (CSTR) is one of the hardest benchmark problems in process control because the titration curve becomes sharply nonlinear near the equivalence point. This project trains a Physics-Informed Neural Network (PINN) on the Hensen-Seborg pH neutralization model, embedding the governing ODEs directly into the loss function so the network learns in a way that is consistent with the physics rather than purely from data.

The trained PINN is then exported and deployed as the internal prediction model of an NMPC controller in MATLAB/Simulink.

## System

**States:** `Wa` (acid reagent invariant), `Wb` (base reagent invariant)  
**Algebraic output:** pH, solved implicitly from the charge balance equation  
**Manipulated input:** `u1` — base flow rate (ml/s)  
**Disturbance input:** `u3` — acid feed flow rate (ml/s)

The pH is not a state; it is computed from `[Wa, Wb]` through the Hensen-Seborg charge balance. The PINN predicts the next-step states, and pH is recovered algebraically.

## PINN Architecture

| Component | Detail |
|-----------|--------|
| Inputs | `[Wa_k, Wb_k, u1_k]` at time step k |
| Outputs | `[Wa_{k+1}, Wb_{k+1}]` predicted next-step states |
| Architecture | Fully connected neural network with tanh activations |
| Physics loss | Normalized ODE residuals |
| Total loss | Data loss + λ × Physics loss |
| Framework | PyTorch |

The physics loss is normalized using the detached RMS of the physical derivative, which helps keep the physics term balanced during training and improves convergence stability.

## Results

### Test Set Performance

| Variable | R² | RMSE | MAE |
|----------|----|------|-----|
| pH | 0.9962 | 0.149 | 0.092 |
| Wₐ | 0.9929 | 4e-5 | 3e-5 |
| W_b | 0.9962 | 1e-5 | <1e-5 |

### One-Step-Ahead Open-Loop Test

| Variable | R² | RMSE |
|----------|----|------|
| pH | 0.9665 | 0.477 |
| Wₐ | 0.9588 | 1.1e-4 |
| W_b | 0.9715 | 2e-5 |

### One-Step-Ahead Validation

| Variable | R² | RMSE |
|----------|----|------|
| pH | 0.8575 | 1.052 |
| Wₐ | 0.8720 | 2.4e-4 |
| W_b | 0.8827 | 4e-5 |

Training converged cleanly in about 300 epochs with no obvious overfitting, and the training and validation loss curves remained closely aligned.

## Repository Structure

```bash
pH_PINN/
├── python/
│   ├── config.py
│   ├── preprocess.py
│   ├── model.py
│   ├── physics.py
│   ├── train.py
│   ├── test.py
│   ├── validate.py
│   ├── pH_train.csv
│   ├── pinn_best.pt
│   ├── pinn_weights.json
│   └── results/
├── matlab/
│   ├── load_pinn.m
│   ├── nmpc_pH.m
│   └── simulink/
└── README.md
```

## Usage

### Training

```bash
cd python/
python train.py
```

### Evaluation

```bash
python test.py
python validate.py
```

### Export weights for MATLAB

```bash
python export_weights.py
```

## Context

This project is part of a broader undergraduate thesis on PINN-based NMPC for chemical process systems. The pH neutralization system is the primary case study because of its strong nonlinear behavior and its importance as a benchmark in process control.

## References

- Hensen, M. A., & Seborg, D. E. (1994). Adaptive nonlinear control of a pH neutralization process. IEEE Transactions on Control Systems Technology, 2(3), 169–182.
- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics, 378, 686–707.

## Author

**Emmanuel Alao**  
Department of Chemical Engineering, Obafemi Awolowo University  
BSc. Final Year — Process Systems Engineering  
[github.com/Makks7](https://github.com/Makks7)
