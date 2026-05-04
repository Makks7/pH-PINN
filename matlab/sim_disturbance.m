% sim_disturbance.m — Disturbance rejection with offset-free NMPC
% Method: Output disturbance estimation (offset-free NMPC)
clear; clc;

%% Load PINN
pinn = load_pinn('pinn_weights.json');

%% Load data for initial conditions
data = readmatrix('../pH_train.csv');

%% Simulation parameters
T_sim = 300;
N = 10;
Q = 100;
R = 0.05;
u_min = 10;
u_max = 20;

%% Fixed setpoint
pH_sp_true = 7.0;
pH_sp = pH_sp_true * ones(T_sim, 1);

%% Disturbance: acid surge at step 150
Wa_disturbance = -0.0004;   % mol
disturbance_step = 150;

%% Offset-free correction: first-order disturbance estimator
% d_hat = (1-Kd)*d_hat + Kd*(pH_measured - pH_predicted)
Kd = 0.2;          % filter gain (0.1-0.3 typical)
d_hat = 0.0;

%% Initial conditions
Wa = data(1, 3);
Wb = data(1, 4);
u_prev = data(1, 2);
pH_meas = data(1, 5);

%% Storage
pH_log = zeros(T_sim, 1);
u_log = zeros(T_sim, 1);
sp_log = zeros(T_sim, 1);
d_log = zeros(T_sim, 1);
pH_log(1) = pH_meas;
u_log(1) = u_prev;
sp_log(1) = pH_sp_true;

%% NMPC loop
fprintf('Running disturbance rejection (offset-free NMPC)...\n');
t0 = tic;

for t = 2:T_sim

    % --- Inject disturbance ---
    if t == disturbance_step
        Wa = Wa + Wa_disturbance;
        fprintf('  [!] Acid disturbance injected at step %d\n', t);
    end

    % --- Predict pH at current state with previous control ---
    y_pred = pinn_predict(pinn, Wa, Wb, u_prev);
    pH_pred = y_pred(3);

    % --- Update disturbance estimate (filtered) ---
    d_hat = (1 - Kd) * d_hat + Kd * (pH_meas - pH_pred);
    d_hat = max(-1.0, min(1.0, d_hat));   % clamp

    % --- Corrected setpoint for optimizer ---
    pH_sp_corrected = pH_sp(t) - d_hat;

    % --- NMPC optimization ---
    u_opt = nmpc_pinn(pinn, Wa, Wb, pH_sp_corrected, u_prev, ...
                      N, Q, R, u_min, u_max);

    % --- Apply control to plant (PINN simulation) ---
    y = pinn_predict(pinn, Wa, Wb, u_opt);

    % --- Update states ---
    Wa = y(1);
    Wb = y(2);
    pH_meas = y(3);
    u_prev = u_opt;

    % --- Log ---
    pH_log(t) = pH_meas;
    u_log(t) = u_opt;
    sp_log(t) = pH_sp_true;
    d_log(t) = d_hat;

    if mod(t, 50) == 0
        fprintf('  Step %d/%d | pH = %.3f | u = %.2f | d_hat = %.4f\n', ...
                t, T_sim, pH_meas, u_opt, d_hat);
    end
end

elapsed = toc(t0);
fprintf('Done. Average solve time: %.3f s/step\n', elapsed/(T_sim-1));

%% Metrics (post-disturbance)
error_post = pH_log(disturbance_step:end) - pH_sp_true;
iae = sum(abs(error_post));
ise = sum(error_post.^2);
max_err = max(abs(error_post));

settled = find(abs(error_post) < 0.1, 1, 'first');
if isempty(settled)
    settled_str = 'Did not settle';
else
    settled_str = sprintf('%d steps', settled);
end

fprintf('\n=== Disturbance Rejection ===\n');
fprintf('IAE  : %.4f\n', iae);
fprintf('ISE  : %.4f\n', ise);
fprintf('Max |e|: %.4f pH\n', max_err);
fprintf('Settling time: %s\n', settled_str);

%% Plot
figure('Name', 'Disturbance Rejection', 'Position', [100 100 1000 800]);

subplot(4,1,1);
plot(1:T_sim, pH_log, 'b-', 'LineWidth', 1.5); hold on;
plot(1:T_sim, sp_log, 'r--', 'LineWidth', 1.5);
xline(disturbance_step, 'k--', 'Disturbance');
ylabel('pH'); title('pH Response — Offset-Free NMPC');
legend('Controlled pH', 'Setpoint', 'Location', 'best');
grid on;

subplot(4,1,2);
stairs(1:T_sim, u_log, 'g-', 'LineWidth', 1.5);
xline(disturbance_step, 'k--');
ylabel('u_1 (ml/s)'); title('Control Input');
grid on;

subplot(4,1,3);
plot(1:T_sim, pH_log - sp_log, 'm-', 'LineWidth', 1.2);
xline(disturbance_step, 'k--'); yline(0, 'k--');
yline(0.1, 'b:'); yline(-0.1, 'b:');
ylabel('Error'); title('Tracking Error');
grid on;

subplot(4,1,4);
plot(1:T_sim, d_log, 'Color', [0.8 0.4 0], 'LineWidth', 1.2);
xline(disturbance_step, 'k--'); yline(0, 'k--');
ylabel('d_{hat}'); title('Disturbance Estimate');
xlabel('Time step');
grid on;

sgtitle('PINN-NMPC Disturbance Rejection (Offset-Free via Output Estimation)');