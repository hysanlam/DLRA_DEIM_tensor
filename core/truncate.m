function T=truncate(X,r)

U1=X.U{1};
U2=X.U{2};
U3=X.U{3};
S=X.core;

[Q1,R1]=qr(U1,'econ');
[Q2,R2]=qr(U2,'econ');
[Q3,R3]=qr(U3,'econ');

S=ttm(S, {R1,R2,R3}, [1 2 3]);

[U_temp_1,~,~]=svd(tenmat(S,1).data,"econ");
[U_temp_2,~,~]=svd(tenmat(S,2).data,"econ");
[U_temp_3,~,~]=svd(tenmat(S,3).data,"econ");

U_temp_1=U_temp_1(:,1:r(1));
U_temp_2=U_temp_2(:,1:r(2));
U_temp_3=U_temp_3(:,1:r(3));
S_new=ttm(S, {U_temp_1',U_temp_2',U_temp_3'}, [1 2 3]);

T=ttensor(S_new,{Q1*U_temp_1,Q2*U_temp_2,Q3*U_temp_3});

end