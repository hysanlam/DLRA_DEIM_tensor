function Y=projected_rk3_deim(Y,h,F_eval_entries,F_eval,r,deim_solver)

    Y1=truncate(add_tucker_tensors(Y,h./3*projected_F_deim(Y,F_eval_entries,F_eval,deim_solver),[1 2 3]),r);
    Y2=truncate(add_tucker_tensors(Y,2*h./3*projected_F_deim(Y1,F_eval_entries,F_eval,deim_solver),[1 2 3]),r);
    temp=add_tucker_tensors(Y,0.25*h*projected_F_deim(Y,F_eval_entries,F_eval,deim_solver),[1 2 3]);
    
    Y=truncate(add_tucker_tensors(temp,3/4*h*projected_F_deim(Y2,F_eval_entries,F_eval,deim_solver),[1 2 3]),r);
    
end