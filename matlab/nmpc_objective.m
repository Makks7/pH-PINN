function J = nmpc_objective(u_seq, pinn, Wa0, Wb0, pH_sp, N, Q, R, u_prev)
    Wa = Wa0;  Wb = Wb0;
    J = 0;
    for k = 1:N
        y = pinn_predict(pinn, Wa, Wb, u_seq(k));
        pH = y(3);
        J = J + Q * (pH - pH_sp)^2;
        if k == 1
            J = J + R * (u_seq(k) - u_prev)^2;
        else
            J = J + R * (u_seq(k) - u_seq(k-1))^2;
        end
        Wa = y(1);  Wb = y(2);
    end
end