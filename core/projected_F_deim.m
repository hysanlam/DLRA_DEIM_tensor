function P=projected_F_deim(X,F,G,deim_solver)

U1=X.U{1};
U2=X.U{2};
U3=X.U{3};
n=size(U1,1);
np=size(U1,2);
if deim_solver=="QDEIM" 
    [ p1 ,U11] = q_deim( U1 );
    [ p2 ,U22] = q_deim( U2 );
    [ p3 ,U33] = q_deim( U3 );
elseif deim_solver=="ARP" %ARP only works for real case
    [p1,U11]=arp( U1, 1);
    [p2,U22]=arp( U2,1 );
    [p3,U33]=arp( U3,1);
elseif deim_solver=="SRRQR" %SRRQR only works for real case
    [~,~,p1]=sRRQR_rank( U1',2, np );
    [~,~,p2]=sRRQR_rank( U2',2, np );
    [~,~,p3]=sRRQR_rank( U3',2, np );
    p1=p1(1:np);
    p2=p2(1:np);
    p3=p3(1:np);
    U11=U1*pinv(U1(p1,:)); %used svd in this case.
    U22=U2*pinv(U2(p2,:));
    U33=U3*pinv(U3(p3,:));
end
S=X.core;
temp_s_1=tenmat(S,1);
[Q_S_1,~]=qr((temp_s_1.data)','econ');
temp_s_2=tenmat(S,2);
[Q_S_2,~]=qr((temp_s_2.data)','econ');
temp_s_3=tenmat(S,3);
[Q_S_3,~]=qr((temp_s_3.data)','econ');


temp_s_1(:,:)=Q_S_1';
temp_s_1=tensor(temp_s_1);
temp_k=Q_S_1'*kron(U3(p3,:).',U2(p2,:).');


if deim_solver=="SRRQR"
    [~,~,sub_sample_1]=sRRQR_rank( orth(temp_k')',2, np );
    sub_sample_1=sub_sample_1(1:np);

elseif deim_solver=="ARP"
    sub_sample_1=arp( orth(temp_k') );
    sub_sample_1=sub_sample_1(1:np);
elseif deim_solver=="QDEIM"
    [sub_sample_1,~] = q_deim( orth(temp_k') );
    sub_sample_1=sub_sample_1(1:np);
end

F_1=tenmat(F(X,1:n,p2,p3),1).data;
F_1=F_1(:,sub_sample_1);

U1_dot=F_1-U11*F_1(p1,:);
temp_k=temp_k(:,sub_sample_1);
U1_dot=U1_dot/(temp_k);


temp_s_2(:,:)=Q_S_2';
temp_s_2=tensor(temp_s_2)';
temp_k=Q_S_2'*kron(U3(p3,:).',U1(p1,:).');
if deim_solver=="SRRQR"
    [~,~,sub_sample_2]=sRRQR_rank( orth(temp_k')',2, np );
    sub_sample_2=sub_sample_2(1:np);

elseif deim_solver=="ARP"
    sub_sample_2=arp( orth(temp_k'));
    sub_sample_2=sub_sample_2(1:np);
elseif deim_solver=="QDEIM"
    [sub_sample_2,~] = q_deim( orth(temp_k') );
    sub_sample_2=sub_sample_2(1:np);
end

F_2=tenmat(F(X,p1,1:n,p3),2).data;
F_2=F_2(:,sub_sample_2);
U2_dot=F_2-U22*F_2(p2,:);
temp_k=Q_S_2'*kron(U3(p3,:).',U1(p1,:).');
temp_k=temp_k(:,sub_sample_2);
U2_dot=U2_dot/(temp_k);



temp_s_3(:,:)=Q_S_3';
temp_s_3=tensor(temp_s_3);
temp_k=Q_S_3'*kron(U2(p2,:).',U1(p1,:).');
if deim_solver=="SRRQR"
    [~,~,sub_sample_3]=sRRQR_rank( orth(temp_k')',2, np );
    sub_sample_3=sub_sample_3(1:np);

elseif deim_solver=="ARP"
    sub_sample_3=arp(orth(temp_k') );
    sub_sample_3=sub_sample_3(1:np);
elseif deim_solver=="QDEIM"
    [sub_sample_3,~] = q_deim( orth(temp_k') );
    sub_sample_3=sub_sample_3(1:np);
end

F_3=tenmat(F(X,p1,p2,1:n),3).data;
F_3=F_3(:,sub_sample_3);
U3_dot=F_3-U33*F_3(p3,:);
temp_k=temp_k(:,sub_sample_3);
U3_dot=U3_dot/(temp_k);


P1=add_tucker_tensors(ttensor(F(X,p1,p2,p3),{U11,U22,U33}),ttensor(temp_s_1,{U1_dot,U2,U3}),[1 2 3]);
P2=add_tucker_tensors(ttensor(temp_s_3,{U1,U2,U3_dot}),ttensor(temp_s_2,{U1,U2_dot,U3}),[1 2 3]);
P=add_tucker_tensors(P1,P2,[1 2 3]);


end






