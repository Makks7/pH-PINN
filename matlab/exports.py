import torch, json

checkpoint = torch.load(r"C:\Users\USER PC\Desktop\pH_PINN\pinn_best.pt", 
                        map_location="cpu", 
                        weights_only=False)

# Extract weights
weights = {}
for name, param in checkpoint['model_state'].items():
    weights[name] = param.numpy().tolist()

# Package everything MATLAB needs
export = {
    "weights": weights,
    "scaler_X_min": checkpoint['scaler_X_min'].tolist(),
    "scaler_X_max": checkpoint['scaler_X_max'].tolist(),
    "scaler_Y_min": checkpoint['scaler_Y_min'].tolist(),
    "scaler_Y_max": checkpoint['scaler_Y_max'].tolist(),
    "pH_min": float(checkpoint['pH_min']),
    "pH_max": float(checkpoint['pH_max']),
    "dt_eff": float(checkpoint['dt_eff']),
    "model_config": checkpoint['model_config']
}

with open(r"C:\Users\USER PC\Desktop\pH_PINN\matlab\pinn_weights.json", "w") as f:
    json.dump(export, f, indent=2)

print("Exported keys:", list(export.keys()))
print("\nWeight layer keys:")
for k in weights.keys():
    print(f"  {k}")
print("\nModel config:", checkpoint['model_config'])
print("dt_eff:", checkpoint['dt_eff'])
print("Done.")