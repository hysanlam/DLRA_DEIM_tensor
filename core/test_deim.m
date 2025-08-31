rng(123435)
global F
X=randn(100,100,100);%+1i*randn(100,100,100);
X=hosvd(tensor(X),1e-10,'ranks',[10,10,10]);

Y=randn(100,100,100);%+1i*randn(100,100,100);
Y=hosvd(tensor(Y),1e-10,'ranks',[10,10,10]);
F=full(X)+1e-4*full(Y);
F=hosvd(tensor(F),1e-10,'ranks',[10,10,10]);
F=full(F);
tic
 norm(full(projected_F(X,@F_eval))-F)
 toc
% norm(full(projected_F(projected_F(X,@F_eval),@F_eval))-projected_F(X,@F_eval))
% norm (full(projected_F(X,@F_eval))-full(projected_F_deim(projected_F(X,@F_eval),@F_eval_entries,@F_eval)))
 tic;
 norm(full(projected_F_deim(X,@F_eval_entries,@F_eval,"ARP"))-F)
 toc
 
% norm(full(projected_F_deim(projected_F_deim(X,@F_eval_entries,@F_eval),@F_eval_entries,@F_eval))-projected_F_deim(X,@F_eval_entries,@F_eval))

%norm(full(projected_F_deim(projected_F(X,@F_eval),@F_eval_entries,@F_eval))-projected_F(X,@F_eval))

function F_val=F_eval_entries(X,p1,p2,p3)
global F
temp=full(F);
F_val=tensor(temp(p1,p2,p3));

end


function F_val=F_eval(X)
global F
F_val=F;

end