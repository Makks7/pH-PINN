function y = pinn_predict(pinn, Wa_k, Wb_k, u1_k)

    % --- Normalize input ---
    x = [Wa_k; Wb_k; u1_k];
    x_norm = (x - pinn.X_min(:)) ./ (pinn.X_max(:) - pinn.X_min(:));

    % --- Forward pass ---
    h = tanh(pinn.W0 * x_norm + pinn.b0(:));
    h = tanh(pinn.W2 * h      + pinn.b2(:));
    h = tanh(pinn.W4 * h      + pinn.b4(:));
    
    % CRITICAL: Apply Sigmoid to match PyTorch nn.Sigmoid() output layer
    y_norm = 1 ./ (1 + exp(-(pinn.W6 * h + pinn.b6(:))));
    
    % y_norm is now [0,1] for ALL outputs (Wa, Wb, pH)

    % --- Denormalize outputs ---
    Wa_next = y_norm(1) * (pinn.Y_max(1) - pinn.Y_min(1)) + pinn.Y_min(1);
    Wb_next = y_norm(2) * (pinn.Y_max(2) - pinn.Y_min(2)) + pinn.Y_min(2);
    pH_next = y_norm(3) * (pinn.pH_max - pinn.pH_min) + pinn.pH_min;

    y = [Wa_next; Wb_next; pH_next];
end