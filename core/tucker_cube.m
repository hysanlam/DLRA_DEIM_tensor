function X3 = tucker_cube(X)
%TUCKER_CUBE_FAST  One-shot X.^3 in Tucker form (no r^2 intermediate).
%
%   X3 = tucker_cube_fast( X )
%   Same input/output as tucker_cube, but builds the cube directly:
%     • each factor gets every triple column-wise product of the
%       original factor (→ size n_n × r^3)
%     • core is triple Kronecker of vec(G) then reshaped/perm’d
%
%   Useful when r is modest but r^2 would already be large.
%
G  = X.core;     U = X.U;
r1 = size(G,1);  r2 = size(G,2);  r3 = size(G,3);

% --- factors -----------------------------------------------------------
% mode-1
U1 = U{1};  n1 = size(U1,1);
U1c = reshape(U1,[n1 r1 1 1]) .* reshape(U1,[n1 1 r1 1]) .* reshape(U1,[n1 1 1 r1]);
U1c = reshape(U1c, [n1, r1^3]);

% mode-2
U2 = U{2};  n2 = size(U2,1);
U2c = reshape(U2,[n2 r2 1 1]) .* reshape(U2,[n2 1 r2 1]) .* reshape(U2,[n2 1 1 r2]);
U2c = reshape(U2c, [n2, r2^3]);

% mode-3
U3 = U{3};  n3 = size(U3,1);
U3c = reshape(U3,[n3 r3 1 1]) .* reshape(U3,[n3 1 r3 1]) .* reshape(U3,[n3 1 1 r3]);
U3c = reshape(U3c, [n3, r3^3]);

% --- core --------------------------------------------------------------
g = G(:);                              % length r1*r2*r3
g3 = kron(g, kron(g,g));               % length (r1*r2*r3)^3
G9 = reshape(g3, [r1 r2 r3  r1 r2 r3  r1 r2 r3]); % 9-D

% interleave modes (1 4 7)(2 5 8)(3 6 9)
Gperm = permute(G9, [1 4 7  2 5 8  3 6 9]);
Gcube = reshape(Gperm, [r1^3, r2^3, r3^3]);

% --- pack --------------------------------------------------------------
X3 = ttensor(tensor(Gcube), {U1c, U2c, U3c});
end