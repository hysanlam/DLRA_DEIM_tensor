function Y=projected_rk2_deim(Y,h,F_eval_entries,F_eval,r,deim_solver)

    Y1=truncate(add_tucker_tensors(Y,h*projected_F_deim(Y,F_eval_entries,F_eval,deim_solver),[1 2 3]),r);
    temp=add_tucker_tensors(Y,0.5*h*projected_F_deim(Y,F_eval_entries,F_eval,deim_solver),[1 2 3]);
    
    Y=truncate(add_tucker_tensors(temp,0.5*h*projected_F_deim(Y1,F_eval_entries,F_eval,deim_solver),[1 2 3]),r);
    
end