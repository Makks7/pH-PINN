function u_opt = nmpc_pinn(pinn, Wa, Wb, pH_sp, u_prev, N, Q, R, u_min, u_max)
% Single NMPC optimization step using fmincon

    u0 = u_prev * ones(N, 1);
    A = [];  b = [];  Aeq = [];  beq = [];
    lb = u_min * ones(N, 1);
    ub = u_max * ones(N, 1);
    
    options = optimoptions('fmincon', ...
        'Display', 'off', ...
        'Algorithm', 'sqp', ...
        'MaxIterations', 100, ...
        'StepTolerance', 1e-4);
    
    [u_seq_opt, ~, exitflag] = fmincon(...
        @(u) nmpc_objective(u, pinn, Wa, Wb, pH_sp, N, Q, R, u_prev), ...
        u0, A, b, Aeq, beq, lb, ub, [], options);
    
    if exitflag <= 0
        warning('fmincon did not converge. Using previous control.');
        u_opt = u_prev;
    else
        u_opt = u_seq_opt(1);
    end
end