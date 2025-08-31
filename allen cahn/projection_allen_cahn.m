addpath('../core')
tic
n = 150;                        % Number of points in each dimension
x_range = linspace(0, 2*pi, n); % Grid points in [0, 2π]
dx=x_range(2)-x_range(1);

[X1, X2, X3] = ndgrid(x_range, x_range, x_range); % 3D grid

exp_term = exp(-tan(X1).^2) + exp(-tan(X2).^2) + exp(-tan(X3).^2);
denominator = 1 + exp(abs(csc(-X1/2))) + exp(abs(csc(-X2/2))) + exp(abs(csc(-X3/2)));
Y_inital =u0(X1, X2, X3);


F_full=@(X) F_eval_full(X,dx);
ref_h=1e-3;
for i=1:(1/ref_h)
    Y_inital=RK4(Y_inital,F_full,ref_h);
    i
end

T=14;



Y_RK4=Y_inital;

 % for i=1:T./h   
 %     Y_RK4=RK4(Y_RK4,F_full,h);
 %     i
 % end


error_rk2=[];
error_rk2_table=[];
error_euler_table=[];
error_projection=[];
error_diemprojection=[];

error_projection_high=[];
error_diemprojection_high=[];
error_t=[];
time =logspace(-1.3,-2.4,6);
time=0.001
dt=T./round(T./time);
for ranks=[15]
    r=[ranks ranks ranks];
    error_rk2=[];
    error_euler=[];
    error_trun_rk4=[];
    error_rk2_high=[];
    error_euler_high=[];
    error_trun_rk4_high=[];
    for h=dt
        %
        Y_RK4=Y_inital;
        Y_projected_rk2=hosvd(tensor(Y_inital),1e-14,'ranks',r);
        Y_projected_rk2_high=hosvd(tensor(Y_inital),1e-14,'ranks',r+5);
        Y_projected_euler=hosvd(tensor(Y_inital),1e-14,'ranks',r);
        Y_projected_euler_high=hosvd(tensor(Y_inital),1e-14,'ranks',r+5);
        for i=1:T./h
            
             
            for ref_i=1:(h./ref_h)
                Y_RK4=RK4(Y_RK4,F_full,ref_h);
            end
            Y_projected_rk2=projected_rk2_deim(Y_projected_rk2,h,@(X,U,J,K) F_eval_entries(X,dx,U,J,K),@F_eval,r,"SRRQR");
            Y_projected_rk2_high=projected_rk2_deim(Y_projected_rk2_high,h,@(X,U,J,K) F_eval_entries(X,dx,U,J,K),@F_eval,r+5,"SRRQR");
 
            rk4_norm=norm(Y_RK4,'fro');
            Tr=hosvd(tensor(Y_RK4),1e-14,'ranks',r);
            error_t=[norm(Y_RK4-full(Y_projected_rk2))./rk4_norm,norm(Y_RK4-full(Tr))./rk4_norm]
            error_rk2=[error_rk2,error_t(1)];
            error_trun_rk4=[error_trun_rk4,error_t(2)];
            F_temp=F_full(Y_RK4);
            error_projection=[error_projection,norm(F_temp-full(projected_F(Tr,F_full)))./norm(F_temp,'fro')];
            error_diemprojection=[error_diemprojection,norm(F_temp-full(projected_F_deim(Tr,@(X,U,J,K) F_eval_entries(X,dx,U,J,K),@F_eval,"SRRQR")))./norm(F_temp,'fro')];
            
            Tr_high=hosvd(tensor(Y_RK4),1e-14,'ranks',r+5);
            error_t=[norm(Y_RK4-full(Y_projected_rk2_high))./rk4_norm,norm(Y_RK4-full(Tr_high))./rk4_norm]
            error_rk2_high=[error_rk2_high,error_t(1)];

            error_trun_rk4_high=[error_trun_rk4_high,error_t(2)];

            error_projection_high=[error_projection_high,norm(F_temp-full(projected_F(Tr_high,F_full)))./norm(F_temp,'fro')];
            error_diemprojection_high=[error_diemprojection_high,norm(F_temp-full(projected_F_deim(Tr_high,@(X,U,J,K) F_eval_entries(X,dx,U,J,K),@F_eval,"SRRQR")))./norm(F_temp,'fro')];
    end

       
    end
    error_euler_table=[error_euler_table;error_euler];
    error_rk2_table=[error_rk2_table; error_rk2];
end

 norm(Y_RK4-full(Y_projected_rk2))
toc

subplot(1,2,1);
x = linspace(0,8,8000);
semilogy(x,error_rk2(1:8000),LineWidth=1.5)

hold on
semilogy(x,error_rk2_high(1:8000),LineWidth=1.5)
semilogy(x,error_trun_rk4(1:8000),LineStyle="--")
semilogy(x,error_trun_rk4_high(1:8000),LineStyle="--")
legend("PRK-DEIM 2 r=15","PRK-DEIM 2 r=20","Best r=15 trucation error","Best r=20 trucation error")
xlabel('time')
ylabel('Relative error')
set(gca,'FontSize',15)
subplot(1,2,2);
semilogy(x,error_diemprojection(1:8000),LineWidth=1.5);hold on;
semilogy(x,error_projection(1:8000),LineWidth=1.5)
semilogy(x,error_diemprojection_high(1:8000),LineWidth=1.5);hold on;
semilogy(x,error_projection_high(1:8000),LineWidth=1.5)
legend("Diem projector r=15","Orthogonal error r=15","Diem projector r=20","Orthogonal error r=20")
ylabel('$\|P-F\|_F/\|F\|_F$',Interpreter='latex')
xlabel('time')
set(gca,'FontSize',15)

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

function F_val=F_eval_full(X,dx)       
        alpha=0.1;
        X=full(X);
        F_val=alpha*applyLaplacian3D(X, dx)+X-X.^3;
end

function D = secondDiffMatrix1D(N,h)

    e = ones(N,1);
    
    % Construct the tridiagonal part using spdiags
    % The three diagonals: -2 on the main diagonal, 1 on the sub- and super-diagonals
    D = spdiags([e -2*e e], [-1 0 1], N, N);
    
    % Impose periodic boundary conditions by setting the top-right and bottom-left elements
    D(1, N) = 1;
    D(N, 1) = 1;
    
end

function L = applyLaplacian3D(X, dx)
    
    Lx = (circshift(X, [1, 0, 0]) - 2 * X + circshift(X, [-1, 0, 0])) / dx^2;
    Ly = (circshift(X, [0, 1, 0]) - 2 * X + circshift(X, [0, -1, 0])) / dx^2;
    Lz = (circshift(X, [0, 0, 1]) - 2 * X + circshift(X, [0, 0, -1])) / dx^2;
    L = Lx + Ly + Lz;

end


function LX_array = tuckerLaplacianEntries(G, U1, U2, U3, D, I, J, K)
    D1=D;
    D2=D;
    D3=D;

    DU1=D*U1;

    DU2=D*U2;

    DU3=D*U3;


    % Using the Tensor Toolbox, convert the core to a tensor object:
    G_tensor = tensor(G);

    % Contract the core with the factor slices along each mode:
    % The ttm (tensor-times-matrix) function performs mode-n multiplication.
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

function R = row_kron(A, B)

    [M, n] = size(A);
    [M2, p] = size(B);
    if M ~= M2
        error('A and B must have the same number of rows.');
    end
    R = zeros(M, n * p);
    for i = 1:M
        R(i,:) = kron(A(i,:), B(i,:));
    end
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


