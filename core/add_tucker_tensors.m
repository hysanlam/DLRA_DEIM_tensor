function [result] = add_tucker_tensors(X,Y,mode)
% ADD_TUCKER_TENSORS - Add two Tucker tensors and return a new Tucker decomposition.
U1=X.U;
S1=X.core;
U2=Y.U;
S2=Y.core;
P1 = cell(1, 3);
P2 = cell(1, 3);
U_new=X.U;

for k = mode
    [U_new{k},R]= qr([U1{k}, U2{k}],0); % Concatenating along second dimension

    P1{k} = ((U_new{k})'*U1{k}); % Projection matrices
    P2{k} = ((U_new{k})'*U2{k});

end
S1_new = ttm(S1, P1(mode), mode);
S2_new = ttm(S2, P2(mode), mode);

% Sum core tensors
G_new = full(S1_new) + full(S2_new);
result=ttensor(G_new,U_new);

% Z=full(X)+full(Y);
% result=hosvd(Z,1e-9);
end