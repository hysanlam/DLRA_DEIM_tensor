
U = randn(6,4);              % square test matrix
[R,p] = rowPivotQR_Tall(U)
p
[Q,R,P]=qr(U','vector')


function [U,piv] = rowPivotQR_Tall(U)
% rowPivotQR_Tall   Three-step QRCP-on-U^T algorithm for tall U (n×r, n>r).
%
%   [U,piv] = rowPivotQR_Tall(U) runs r steps of
%     1) select row i_k of U having maximal 2-norm,
%     2) x = U(i_k,:)'/norm(U(i_k,:),2),
%     3) U = U*(I_r - x*x'),
%   and returns the final U and the pivot list piv (length r).

[n,r] = size(U);
piv   = zeros(r,1);

for k = 1:r
    % --- 1) select pivot row among all rows (old pivots have become zero) ---
    rowNorms = sqrt(sum(U.^2,2));       % n×1 of current row norms
    rowNorms
    [~, idx] = max(rowNorms);           % pick the largest
    piv(k)   = idx;
    
    % --- 2) form x from that row and normalize ---
    x = U(idx,:).';                     % r×1 column
    nx = norm(x);
    if nx == 0
        warning('Zero pivot encountered at step %d.', k);
        return
    end
    x = x / nx;
    
    % --- 3) project U ← U*(I_r - x*x') ---
    U = U - U*(x*x.');
end
end