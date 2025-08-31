function P=projected_F(X,F)
U1=X.U{1};
U2=X.U{2};
U3=X.U{3};
S=X.core;
F=F(X);
S_dot=ttm(F, {U1',U2',U3'}, [1 2 3]);
temp=ttm(F, {U2',U3'}, [2 3]);
U1_dot=tenmat(temp,1)-U1*(tenmat(ttm(temp,U1',[1]),1));
U1_dot=U1_dot.data/tenmat(S,1).data;

temp=ttm(F, {U1',U3'}, [1 3]);
U2_dot=tenmat(temp,2)-U2*(tenmat(ttm(temp,U2',[2]),2));
U2_dot=U2_dot.data/tenmat(S,2).data;

temp=ttm(F, {U1',U2'}, [1 2]);
U3_dot=tenmat(temp,3)-U3*(tenmat(ttm(temp,U3',[3]),3));
U3_dot=U3_dot.data/tenmat(S,3).data;

P1=add_tucker_tensors(ttensor(S_dot,{U1,U2,U3}),ttensor(S,{U1_dot,U2,U3}),[1]);
P2=add_tucker_tensors(ttensor(S,{U1,U2,U3_dot}),ttensor(S,{U1,U2_dot,U3}),[2 3]);
P=add_tucker_tensors(P1,P2,[1 2 3]);



end