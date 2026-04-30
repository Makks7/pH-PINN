import torch
import torch.nn as nn
from config import MODEL


class PINN(nn.Module):
    """
    MLP for one-step-ahead prediction of Hensen-Seborg system.

    Input  : [Wa_k, Wb_k, u_k]                    normalized [0,1]
    Output : [Wa_next, Wb_next, pH_next]           normalized [0,1]

    Direct pH output eliminates reliance on the algebraic bisection
    solver during training, which fails on the steep alkaline S-curve.
    Physics is enforced via the ODE residual loss term in train.py.
    """

    def __init__(self):
        super().__init__()
        act_map = {"tanh": nn.Tanh, "relu": nn.ReLU, "silu": nn.SiLU}
        Act     = act_map[MODEL["activation"]]

        layers = []
        in_dim = MODEL["input_dim"]
        for h in MODEL["hidden_layers"]:
            layers += [nn.Linear(in_dim, h), Act()]
            in_dim  = h
        # Sigmoid keeps all outputs in [0,1] matching normalization
        layers += [nn.Linear(in_dim, MODEL["output_dim"]), nn.Sigmoid()]

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """x: [B, 3] -> y: [B, 3]  (Wa_norm, Wb_norm, pH_norm)"""
        return self.net(x)


def model_summary(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] Layers     : {MODEL['hidden_layers']}  |  act: {MODEL['activation']}")
    print(f"[model] Output     : [Wa_next, Wb_next, pH_next]  (all normalized)")
    print(f"[model] Parameters : {total:,} total  |  {trainable:,} trainable")