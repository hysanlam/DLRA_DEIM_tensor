function Y=projected_euler_deim(Y,h,F_eval_entries,F_eval,r,deim_solver)

    Y=truncate(add_tucker_tensors(Y,h*projected_F_deim(Y,F_eval_entries,F_eval,deim_solver),[1 2 3]),r);

end