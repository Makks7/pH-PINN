% test_pinn.m — sanity check before NMPC

pinn = load_pinn('pinn_weights.json');

%% Test 1: Manual steady-state-ish operating point
fprintf('=== Test 1: Manual inputs ===\n');
Wa_k = -0.0003;
Wb_k =  0.0006;
u1_k =  15.0;

y = pinn_predict(pinn, Wa_k, Wb_k, u1_k);

fprintf('Wa_next : %.6f mol\n', y(1));
fprintf('Wb_next : %.6f mol\n', y(2));
fprintf('pH_next : %.4f\n\n',   y(3));

%% Test 2: Real data from training set (CSV is [time, u, Wa, Wb, pH])
fprintf('=== Test 2: Real data row 100 ===\n');
data = readmatrix('../pH_train.csv');

% CSV columns: [time, u, Wa, Wb, pH] — PINN inputs are [Wa, Wb, u]
u1_k = data(100, 2);
Wa_k = data(100, 3);
Wb_k = data(100, 4);
pH_actual = data(100, 5);

fprintf('Input from real data:\n');
fprintf('  Wa_k : %.6f\n', Wa_k);
fprintf('  Wb_k : %.6f\n', Wb_k);
fprintf('  u1_k : %.6f\n', u1_k);

y = pinn_predict(pinn, Wa_k, Wb_k, u1_k);

fprintf('\nPINN prediction:\n');
fprintf('  Wa_next : %.6f mol\n', y(1));
fprintf('  Wb_next : %.6f mol\n', y(2));
fprintf('  pH_next : %.4f\n',     y(3));

fprintf('\nActual pH from data: %.4f\n', pH_actual);