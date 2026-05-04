% test_rollout.m — closed-loop multi-step PINN rollout sanity check
% Run this as a script (not a function). Place in same folder as test_pinn.m.

clear; clc;

%% Load PINN
pinn = load_pinn('pinn_weights.json');

%% Load data
data = readmatrix('../pH_train.csv');
% CSV columns: [time, u, Wa, Wb, pH]

%% Rollout parameters
start_idx = 100;      % starting point in data
horizon   = 100;      % number of steps to predict ahead

%% Initialize from real data at start_idx
Wa = data(start_idx, 3);
Wb = data(start_idx, 4);
u_seq = data(start_idx+1 : start_idx+horizon, 2);  % future control inputs (known from data)

%% Storage
Wa_pred = zeros(horizon, 1);
Wb_pred = zeros(horizon, 1);
pH_pred = zeros(horizon, 1);

%% Autoregressive rollout: predictions fed back as next states
for k = 1:horizon
    y = pinn_predict(pinn, Wa, Wb, u_seq(k));
    Wa_pred(k) = y(1);
    Wb_pred(k) = y(2);
    pH_pred(k) = y(3);
    
    % Feedback: use prediction as next input state
    Wa = y(1);
    Wb = y(2);
end

%% Actual values for comparison
Wa_actual = data(start_idx+1 : start_idx+horizon, 3);
Wb_actual = data(start_idx+1 : start_idx+horizon, 4);
pH_actual = data(start_idx+1 : start_idx+horizon, 5);
t = (0:horizon-1) * pinn.dt;

%% Plot
figure('Name', 'PINN Multi-Step Rollout', 'Position', [100 100 900 700]);

subplot(3, 1, 1);
plot(t, pH_actual, 'b-', 'LineWidth', 1.5); hold on;
plot(t, pH_pred, 'r--', 'LineWidth', 1.5);
ylabel('pH'); title('pH — Multi-Step Rollout');
legend('Actual', 'PINN Rollout', 'Location', 'best');
grid on;

subplot(3, 1, 2);
plot(t, Wa_actual, 'b-', 'LineWidth', 1.5); hold on;
plot(t, Wa_pred, 'r--', 'LineWidth', 1.5);
ylabel('W_a (mol)'); title('W_a — Multi-Step Rollout');
legend('Actual', 'PINN Rollout', 'Location', 'best');
grid on;

subplot(3, 1, 3);
plot(t, Wb_actual, 'b-', 'LineWidth', 1.5); hold on;
plot(t, Wb_pred, 'r--', 'LineWidth', 1.5);
ylabel('W_b (mol)'); xlabel('Time (s)');
title('W_b — Multi-Step Rollout');
legend('Actual', 'PINN Rollout', 'Location', 'best');
grid on;

sgtitle('PINN Closed-Loop Rollout Sanity Check');

%% Metrics
rmse_pH = sqrt(mean((pH_pred - pH_actual).^2));
mae_pH  = mean(abs(pH_pred - pH_actual));
fprintf('\n=== Rollout Metrics (horizon = %d steps) ===\n', horizon);
fprintf('pH RMSE : %.4f\n', rmse_pH);
fprintf('pH MAE  : %.4f\n', mae_pH);
fprintf('pH drift: %.4f (pred end - actual end)\n', pH_pred(end) - pH_actual(end));