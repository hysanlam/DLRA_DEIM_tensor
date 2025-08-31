function [J, M] = arp(U, compute_M, seed)
% ARP   Adaptive Randomized Pivoting for CSSP (Alg. 2.1)
%   J = ARP(U) selects r pivots from an n×r orthonormal matrix U.
%   [J, M] = ARP(U, true) also returns M = U * inv(U(J,:)).
%   J = ARP(U, false, seed) lets you seed MATLAB's RNG for reproducibility.
%
% Inputs:
%   U          n×r matrix with orthonormal columns.
%   compute_M  (optional) logical; if true, returns M.  Default false.
%   seed       (optional) integer seed for rng.          
%
% Outputs:
%   J   1×r vector of pivot indices (into rows of U, i.e. columns of original A).
%   M   (if requested) n×r matrix = U * inv(U(J,:)).

    if nargin >= 3 && ~isempty(seed)
        rng(seed);
    end
    if nargin < 2 || isempty(compute_M)
        compute_M = false;
    end

    [n, r] = size(U);
    J      = zeros(1, r);
    Uk     = U;             % working copy

    for k = 1:r
        % 1) sampling probabilities p_j = ||Uk(j,k:r)||^2 / sum(...)
        row_norms_sq  = sum(Uk(:, k:r).^2, 2);
        total_norm_sq = sum(row_norms_sq);
        probs         = row_norms_sq / total_norm_sq;

        % 2) sample one index
        jk         = randsample(n, 1, true, probs);
        J(k)       = jk;

        % 3) build/apply Householder on the right
        x          = Uk(jk, k:r).';               % (r-k+1)×1
        [v, beta]  = house_vec(x);
        if beta ~= 0
            Uk(:, k:r) = apply_H_right(Uk(:, k:r), v, beta);
        end
    end

    if compute_M
        % M = U * inv(U(J,:))
        M = U / U(J, :);
    else
        M = [];
    end
end


function [v, beta] = house_vec(x)
% HOUSE_VEC  compute Householder vector v and scalar beta so that
%   H = I - beta * v * v'  annihilates all but the first entry of x.

    norm_x = norm(x);
    if norm_x < 1e-15
        v = zeros(size(x)); beta = 0;
        return;
    end

    % alpha = sign(x1)*||x||
    alpha = sign(x(1)) * norm_x;
    v     = x;
    v(1)  = v(1) + alpha;

    v_norm_sq = v' * v;
    if v_norm_sq < 1e-15
        v = zeros(size(x)); beta = 0;
    else
        beta = 2 / v_norm_sq;
    end
end


function A_updated = apply_H_right(A, v, beta)
% APPLY_H_RIGHT  applies H = I - beta v v' on the right:  A := A * H'.
% Since H is symmetric, A*H = A - beta*(A*v)*v'.

    Av         = A * v;          % n×1
    A_updated  = A - beta * (Av * v');
end