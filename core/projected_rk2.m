function Y=projected_rk2(Y,h,F_eval,r)

    Y1=truncate(add_tucker_tensors(Y,h*projected_F(Y,F_eval),[1 2 3]),r);
    temp=add_tucker_tensors(Y,0.5*h*projected_F(Y,F_eval),[1 2 3]);
    
    Y=truncate(add_tucker_tensors(temp,0.5*h*projected_F(Y1,F_eval),[1 2 3]),r);
    
end