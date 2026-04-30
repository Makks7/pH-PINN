import os

_ROOT = os.path.dirname(os.path.abspath(__file__))

PATHS = {
    "data":       os.path.join(_ROOT, "pH_train.csv"),
    "model_ckpt": os.path.join(_ROOT, "pinn_best.pt"),
    "results":    os.path.join(_ROOT, "results"),
}

PHYSICS = {
    "V":    2900.0,
    "u3":   16.60,  "u2":   0.55,
    "Wa3": -3.05e-3, "Wa1":  3.00e-3, "Wa2": -3.00e-2,
    "Wb3":  5.00e-5, "Wb1":  0.00,    "Wb2":  3.00e-2,
    "pk1":  6.35,
    "pk2": -10.25,
    "Wa_nom": -4.32e-4,
    "Wb_nom":  5.28e-4,
    "pH_nom":  7.0,
    "dt_raw":  5.0,
    "subsample_step": 2,
}

MODEL = {
    "input_dim":     3,
    "output_dim":    3,              # [Wa_next, Wb_next, pH_next]
    "hidden_layers": [128, 128, 64],
    "activation":    "tanh",
}

TRAIN = {
    "seed":        42,
    "test_frac":   0.20,
    "val_frac":    0.10,
    "batch_size":  512,
    "epochs":      2000,
    "lr":          1e-3,
    "lr_patience": 60,
    "lr_factor":   0.5,
    "early_stop":  200,
    # L = w_state*L_state + w_pH*L_pH + w_phys*L_phys
    # w_phys reduced to 0.005 -- physics regularizes gently, data supervises strongly
    "w_state": 2.0,
    "w_pH":    2.0,
    "w_phys":  0.005,
}