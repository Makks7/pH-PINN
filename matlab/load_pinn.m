function pinn = load_pinn(json_path)
% LOAD_PINN  Load exported PINN weights and scalers from JSON

    raw = jsondecode(fileread(json_path));

    % --- Weights and biases ---
    pinn.W0 = raw.weights.net_0_weight;
    pinn.b0 = raw.weights.net_0_bias;
    pinn.W2 = raw.weights.net_2_weight;
    pinn.b2 = raw.weights.net_2_bias;
    pinn.W4 = raw.weights.net_4_weight;
    pinn.b4 = raw.weights.net_4_bias;
    pinn.W6 = raw.weights.net_6_weight;
    pinn.b6 = raw.weights.net_6_bias;

    % --- Scalers ---
    pinn.X_min = raw.scaler_X_min(:);
    pinn.X_max = raw.scaler_X_max(:);
    
    % scaler_Y is ONLY for Wa, Wb (2 outputs). Do NOT append pH bounds.
    pinn.Y_min = raw.scaler_Y_min(:);   % [2x1]
    pinn.Y_max = raw.scaler_Y_max(:);   % [2x1]
    
    % pH bounds stored separately
    pinn.pH_min = raw.pH_min;
    pinn.pH_max = raw.pH_max;

    % --- Metadata ---
    pinn.dt = raw.dt_eff;

    disp('PINN loaded successfully.');
end