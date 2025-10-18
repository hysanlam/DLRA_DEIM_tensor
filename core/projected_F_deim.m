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

function [ p ] = gpode( U, np )
[~,~,p] = qr(U', 'vector'); % qdeim
p = p(1:size(U,2));         % take points equal to number of basis
for i=length(p)+1:np
    [~, S, W] = svd(U(p, :), 0);
    g = S(end-1, end-1)^2 - S(end, end)^2;
    Ub = W'*U';
    r = g + sum(Ub.^2, 1);
    r = r-sqrt((g+sum(Ub.^2,1)).^2-4*g*Ub(end, :).^2);
    [~, I] = sort(r, 'descend');
    e = 1;
    while any(I(e) == p)
        e = e + 1;
    end
    p(end + 1) = I(e);
end
p = sort(p);
end


function result = accessMode1Columns(getA, p1, p2, I, I1)


L1 = length(p1);    % New mode-2 dimension in B
numCols = numel(I); % Number of columns to extract

% Preallocate the result matrix with size [I1 x numCols]
result = zeros(I1, numCols);

% Loop over each desired column index in I
for colIdx = 1:numCols
    c = I(colIdx);
    % Recover j and k from the column index c using the unfolding mapping:
    j = mod(c - 1, L1) + 1;
    k = floor((c - 1) / L1) + 1;

    % For each row i in the first mode, fill in the value:

    result(1:I1, colIdx) = getA(1:I1, p1(j), p2(k));

end
end

function result = accessMode2Columns(getA, p1, p3, I,I2)

L1 = length(p1);       % Number of rows in B corresponding to the first mode
numCols = numel(I);    % Number of columns to extract
result = zeros(I2, numCols); % Preallocate result matrix

% Loop over each desired column index in I.
for idx = 1:numCols
    c = I(idx);
    % Recover i and k from the unfolding mapping:
    i = mod(c-1, L1) + 1;
    k = floor((c-1)/L1) + 1;

    % Instead of constructing the full subtensor B, we directly extract the
    % vector from the second mode for the appropriate first and third mode indices.
    result(:, idx) = getA(p1(i), 1:I2, p3(k));
end
end

function result = accessMode3Columns(getA, p1, p2, I,I3)

L1 = length(p1);   % New first-mode dimension in B
L2 = length(p2);   % New second-mode dimension in B
numCols = numel(I);

% Preallocate the result matrix F of size [n x numCols]
result = zeros(I3, numCols);

for idx = 1:numCols
    c = I(idx);
    % Recover indices in p1 and p2 from the unfolding mapping:
    i1 = mod(c-1, L1) + 1;
    i2 = floor((c-1)/L1) + 1;
    result(:, idx) = getA(p1(i1), p2(i2), 1:I3);
end
end
