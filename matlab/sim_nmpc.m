% sim_nmpc.m — closed-loop NMPC simulation using PINN
clear; clc;

%% Load PINN
pinn = load_pinn('pinn_weights.json');

%% Load test data for initial conditions and comparison
data = readmatrix('../pH_train.csv');
% Columns: [time, u, Wa, Wb, pH]

%% Simulation parameters
T_sim = 300;           % total simulation steps
start_idx = 1;         % starting point in test data

%% NMPC tuning
N = 10;                % prediction horizon (short to limit drift)
Q = 100;               % pH tracking weight
R = 0.05;              % control move penalty
u_min = 10;            % ml/s
u_max = 20;            % ml/s

%% Setpoint profile (step changes)
pH_sp = 7.0 * ones(T_sim, 1);
pH_sp(100:200) = 6.5;  % step down at t=100
pH_sp(201:T_sim) = 7.2;% step up at t=200

%% Initial conditions from data
Wa = data(start_idx, 3);
Wb = data(start_idx, 4);
u_prev = data(start_idx, 2);
pH_meas = data(start_idx, 5);  % initial pH

%% Storage
pH_log = zeros(T_sim, 1);
u_log = zeros(T_sim, 1);
sp_log = zeros(T_sim, 1);
pH_log(1) = pH_meas;
u_log(1) = u_prev;
sp_log(1) = pH_sp(1);

%% NMPC loop
fprintf('Running NMPC simulation...\n');
t0 = tic;

for t = 2:T_sim
    % NMPC optimization
    u_opt = nmpc_pinn(pinn, Wa, Wb, pH_sp(t), u_prev, N, Q, R, u_min, u_max);
    
    % Apply control to PINN (simulated plant)
    y = pinn_predict(pinn, Wa, Wb, u_opt);
    
    % Update states
    Wa = y(1);
    Wb = y(2);
    pH_meas = y(3);
    u_prev = u_opt;
    
    % Log
    pH_log(t) = pH_meas;
    u_log(t) = u_opt;
    sp_log(t) = pH_sp(t);
    
    % Progress
    if mod(t, 50) == 0
        fprintf('  Step %d/%d  |  pH = %.3f  |  u = %.2f\n', t, T_sim, pH_meas, u_opt);
    end
end

elapsed = toc(t0);
fprintf('Done. Average solve time: %.3f s/step\n', elapsed/(T_sim-1));

%% Plot results
figure('Name', 'NMPC Closed-Loop Simulation', 'Position', [100 100 1000 700]);

% pH tracking
subplot(3, 1, 1);
plot(1:T_sim, pH_log, 'b-', 'LineWidth', 1.5); hold on;
plot(1:T_sim, sp_log, 'r--', 'LineWidth', 1.5);
ylabel('pH');
title('NMPC Setpoint Tracking');
legend('Controlled pH', 'Setpoint', 'Location', 'best');
grid on;

% Control input
subplot(3, 1, 2);
stairs(1:T_sim, u_log, 'g-', 'LineWidth', 1.5);
ylabel('u_1 (ml/s)');
title('Manipulated Input');
grid on;

% Tracking error
subplot(3, 1, 3);
error = pH_log - sp_log;
plot(1:T_sim, error, 'm-', 'LineWidth', 1.2);
yline(0, 'k--');
ylabel('pH error');
xlabel('Time step');
title('Tracking Error');
grid on;

sgtitle('PINN-NMPC Closed-Loop Performance');

%% Metrics
iae = sum(abs(error(2:end)));
ise = sum(error(2:end).^2);
max_error = max(abs(error(2:end)));
fprintf('\n=== NMPC Performance ===\n');
fprintf('IAE  : %.4f\n', iae);
fprintf('ISE  : %.4f\n', ise);
fprintf('Max |e|: %.4f pH\n', max_error);