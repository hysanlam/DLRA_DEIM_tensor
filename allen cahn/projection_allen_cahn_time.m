addpath('../core')
rng(1234);
n = 150;                        % Number of points in each dimension
x_range = linspace(0, 2*pi, n); % Grid points in [0, 2pi]
dx=x_range(2)-x_range(1);

[X1, X2, X3] = ndgrid(x_range, x_range, x_range); % 3D grid

exp_term = exp(-tan(X1).^2) + exp(-tan(X2).^2) + exp(-tan(X3).^2);
denominator = 1 + exp(abs(csc(-X1/2))) + exp(abs(csc(-X2/2))) + exp(abs(csc(-X3/2)));
Y_inital =u0(X1, X2, X3);


F_full=@(X) F_eval_full(X,dx);
F_tucker=@(X) F_eval_tucker(X,dx);

%Time integration with RK4 to final avoid ill conditedioned.
ref_h=1e-3;
for i=1:(1/ref_h)
    Y_inital=RK4(Y_inital,F_full,ref_h);
    i
end

%Time integration with RK4 to final T=1;
T=1;
Y_RK4=Y_inital;

for i=1:T./ref_h
    Y_RK4=RK4(Y_RK4,F_full,ref_h);
    i
end


% Main loop;
time=0.001
dt=T./round(T./time);
error_list_sqdeim=[];
error_list_arp=[];
error_list_proj=[];
time_list_sqdeim=[];
time_list_arp=[];
time_list_proj=[];


for ranks=2:5
    r=[ranks ranks ranks];
    
    Y_projected_rk2_sqdeim=hosvd(tensor(Y_inital),1e-14,'ranks',r);
    Y_projected_rk2_ARP=hosvd(tensor(Y_inital),1e-14,'ranks',r);
    Y_projected_rk2=hosvd(tensor(Y_inital),1e-14,'ranks',r);

    tic
    for i=1:T./dt
        Y_projected_rk2_sqdeim=projected_rk2_deim(Y_projected_rk2_sqdeim,dt,@(X,U,J,K) F_eval_entries(X,dx,U,J,K),@F_eval,r,"SRRQR");
    end
    time_list_sqdeim=[time_list_sqdeim,toc];
    tic
    for i=1:T./dt
        Y_projected_rk2_ARP=projected_rk2_deim(Y_projected_rk2_ARP,dt,@(X,U,J,K) F_eval_entries(X,dx,U,J,K),@F_eval,r,"ARP");

    end
    time_list_arp=[time_list_arp,toc];
    
    tic
    for i=1:T./dt
        Y_projected_rk2=projected_rk2(Y_projected_rk2,dt,F_tucker,r);

    end
    time_list_proj=[time_list_proj,toc];

    error_list_proj=[error_list_proj,norm(Y_RK4-full(Y_projected_rk2))./norm(Y_RK4,'fro')];
    error_list_sqdeim=[error_list_sqdeim,norm(Y_RK4-full(Y_projected_rk2_sqdeim))./norm(Y_RK4,'fro')];
    error_list_arp=[error_list_arp,norm(Y_RK4-full(Y_projected_rk2_ARP))./norm(Y_RK4,'fro')]
end




function F_val=F_eval_full(X,dx)
alpha=0.1;
X=full(X);
F_val=alpha*applyLaplacian3D(X, dx)+X-X.^3;
end

function D = secondDiffMatrix1D(N,h)

e = ones(N,1);
D = spdiags([e -2*e e], [-1 0 1], N, N);
D(1, N) = 1;
D(N, 1) = 1;

end

function L = applyLaplacian3D(X, dx)

Lx = (circshift(X, [1, 0, 0]) - 2 * X + circshift(X, [-1, 0, 0])) / dx^2;
Ly = (circshift(X, [0, 1, 0]) - 2 * X + circshift(X, [0, -1, 0])) / dx^2;
Lz = (circshift(X, [0, 0, 1]) - 2 * X + circshift(X, [0, 0, -1])) / dx^2;
L = Lx + Ly + Lz;

end

function F_val=F_eval_entries(X,dx,I,J,K)
alpha=0.1;
U1=X.U{1};
U2=X.U{2};
U3=X.U{3};
S=X.core;

D=secondDiffMatrix1D(size(U1,1),dx);
laplacian= tuckerLaplacianEntries(S, U1, U2, U3, D, I, J, K);


x_temp=tuckerEntries(S, U1, U2, U3, I, J, K);
X_cube=x_temp-x_temp.^3;


F_val=(alpha./ (dx^2))*laplacian+X_cube;
F_val=tensor(F_val);

end

function LX_array = tuckerLaplacianEntries(G, U1, U2, U3, D, I, J, K)

DU1=D*U1;
DU2=D*U2;
DU3=D*U3;

G_tensor = tensor(G);

sum_ten=add_tucker_tensors(ttensor(G_tensor,{DU1,U2,U3}),ttensor(G_tensor,{U1,DU2,U3}),[1 2 3]);
sum_ten=add_tucker_tensors(sum_ten,ttensor(G_tensor,{U1,U2,DU3}),[1 2 3]);
U1=sum_ten.U{1};
U2=sum_ten.U{2};
U3=sum_ten.U{3};
U_I =U1(I, :);  % Size: [length(I) x R1]
V_J = U2(J, :);  % Size: [length(J) x R2]
W_K = U3(K, :);  % Size: [length(K) x R3]

% Using the Tensor Toolbox, convert the core to a tensor object:

G_tensor = sum_ten.core;
LX_array = ttm(G_tensor, {U_I, V_J, W_K}, [1 2 3]);

end

function X_array = tuckerEntries(G, U1, U2, U3, I, J, K)

U_I = U1(I, :);  % Size: [length(I) x R1]
V_J = U2(J, :);  % Size: [length(J) x R2]
W_K = U3(K, :);  % Size: [length(K) x R3]

% Using the Tensor Toolbox, convert the core to a tensor object:
G_tensor = tensor(G);

% Contract the core with the factor slices along each mode:
% The ttm (tensor-times-matrix) function performs mode-n multiplication.
X_array = ttm(G_tensor, {U_I, V_J, W_K}, [1 2 3]);
end


function val = u0(x1, x2, x3)

val = g(x1, x2, x3) ...
    - g(2*x1, 2*x2, 2*x3) ...
    + g(4*x1, 4*x2, 4*x3) ...
    - g(8*x1, 8*x2, 8*x3);

end

function val = g(x1, x2, x3)

numerator = (exp(-tan(x1).^2) + exp(-tan(x2).^2) + exp(-tan(x3).^2)) ...
    .* sin(x1 + x2 + x3);

denominator = 1 ...
    + exp(abs(1 ./ sin(x1/2))) ...
    + exp(abs(1 ./ sin(x2/2))) ...
    + exp(abs(1 ./ sin(x3/2)));

val = numerator ./ denominator;
end

function F_val = F_eval_tucker(X, dx )

U1=X.U{1};
U2=X.U{2};
U3=X.U{3};
G=X.core;
alpha = 0.1;

% Build 1D periodic Laplacian matrices L1,L2,L3

n1 = size(U1,1);
e1 = ones(n1,1);
L1 = spdiags([ e1  -2*e1  e1 ], [-1 0 1], n1, n1) / dx^2;
L1(1,  n1) = 1/dx^2;   L1(n1, 1) = 1/dx^2;

n2 = size(U2,1);
e2 = ones(n2,1);
L2 = spdiags([ e2  -2*e2  e2 ], [-1 0 1], n2, n2) / dx^2;
L2(1,  n2) = 1/dx^2;   L2(n2, 1) = 1/dx^2;

n3 = size(U3,1);
e3 = ones(n3,1);
L3 = spdiags([ e3  -2*e3  e3 ], [-1 0 1], n3, n3) / dx^2;
L3(1,  n3) = 1/dx^2;   L3(n3, 1) = 1/dx^2;


% Build the Tucker object X_tucker = ttensor(G,{U1,U2,U3})

X_tucker = ttensor( G, {U1, U2, U3} );
LapX1 = ttm( X_tucker, L1, 1 );   % apply L1 on mode–1
LapX2 = ttm( X_tucker, L2, 2 );   % apply L2 on mode–2
LapX3 = ttm( X_tucker, L3, 3 );   % apply L3 on mode–3
Temp1 = add_tucker_tensors(LapX1, LapX2, [1 2 3]);
Temp2 = add_tucker_tensors(Temp1, LapX3, [1 2 3]);
Gtemp = Temp2.core;           % a Tensor‐object
Gtemp = alpha * Gtemp;        % scalar*core is a valid Tensor
LapX   = ttensor(Gtemp, Temp2.U);


Xterm = X_tucker;

X3_uncompressed=tucker_cube(X_tucker);

% Negate the core to represent – X^3
G3_neg = -1 * X3_uncompressed.core;
X3_neg  = ttensor( tensor(G3_neg), X3_uncompressed.U );   % a ttensor object


% Sum up:  LapX  +  Xterm  +  (− X^3),
Sum1 = add_tucker_tensors( LapX,   Xterm,   [1 2 3] );
Sum2 = add_tucker_tensors( Sum1, X3_neg, [1 2 3] );

F_val = Sum2;
end

