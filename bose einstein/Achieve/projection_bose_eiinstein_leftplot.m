addpath('../core')
n = 100;                        % Number of points in each dimension
gamma = 10;
j1 = 75; k1 = 25; l1 = 1; % First peak position
j2 = 25; k2 = 75; l2 = 100; % Second peak position

% Initialize the 3D array
A = zeros(n, n, n);

% Compute values based on the given formula
for j = 1:n
    for k = 1:n
        for l = 1:n
            A(j, k, l) = exp(-1/gamma^2 * ((j - j1)^2 + (k - k1)^2 + (l - l1)^2)) + exp(-1/gamma^2 * ((j - j2)^2 + (k - k2)^2 + (l - l2)^2));
        end
    end
end


Y_inital = A;
F_full=@(X) F_eval_full(X);
h=0.5e-3;
for i=1:20
    Y_inital=RK4(Y_inital,F_full,h);
end


T=1;


h=0.5e-3;
Y_RK4=Y_inital;
Y_euler=Y_inital;
for i=1:T./h
    Y_RK4=RK4(Y_RK4,F_full,h);
end

error_rk2=[];
error_rk2_table=[];
error_rk3=[];
error_rk3_table=[];
error_euler=[];
time =logspace(-1,-3,7);
time=1e-1;
dt=T./round(T./time);
for ranks=[10]
    r=[ranks ranks ranks];
    error_rk2=[];
    for h=dt
        Y_projected_rk3=hosvd(tensor(Y_inital),1e-10,'ranks',r);
        Y_projected_euler=Y_projected_rk3;
        Y_projected_rk2=Y_projected_euler;

        for i=1:T./h
            Y_projected_rk3=projected_rk3_deim(Y_projected_rk3,h,@F_eval_entries,@F_eval,r,"SRRQR");
            %Y_projected_euler=projected_euler_deim(Y_projected_euler,h,@F_eval_entries,@F_eval,r,"SRRQR");
            %Y_projected_rk2=projected_rk2_deim(Y_projected_rk2,h,@F_eval_entries,@F_eval,r,"SRRQR");
        end

        error_rk2=[error_rk2,norm(Y_RK4-full(Y_projected_rk2))];
        error_euler=[error_euler,norm(Y_RK4-full(Y_projected_euler))];
        error_rk3=[error_rk3,norm(Y_RK4-full(Y_projected_rk3))];
    end

end

loglog(dt,error_euler,LineWidth=1.5,Marker='o')
hold on
loglog(dt,error_rk2,LineWidth=1.5,Marker='o')
loglog(dt,error_rk3,LineWidth=1.5,Marker='o')
loglog(dt,100*dt,LineWidth=1,LineStyle="--")
loglog(dt,100*dt.^2,LineWidth=1,LineStyle="--")
loglog(dt,100*dt.^3,LineWidth=1,LineStyle="--")
legend("PRK-DEIM 1","PRK-DEIM 2","PRK-DEIM 3","Slope 1","Slope 2","Slope 3")
xlabel('time')
ylabel('Absolute error')
set(gca,'FontSize',15)



function F_val=F_eval(X)
eps=1e-1;
X=full(X).data;
X_cube=abs((X.^2)).*X;

F_val=1i./2*compute_L(X)+eps*X_cube;
F_val=hosvd(tensor(F_val),1e-16);

end

function F_val=F_eval_full(X)

eps=1e-1;
X_cube=abs((X.^2)).*(X);

F_val=1i./2*compute_L(X)+eps*X_cube;

end

function F_val=F_eval_entries(X,p1,p2,p3)
eps=1e-1;
X=full(X).data;
X_cube=abs((X(p1,p2,p3).^2)).*X(p1,p2,p3);

F_val=1i./2*compute_L_entries(X,p1,p2,p3)+eps*X_cube;
F_val=tensor(F_val);

end


function L = compute_L(A)

L = zeros(size(A));

% Loop over all valid j,k,l from 1 to 100
for j = 1:100
    for k = 1:100
        for l = 1:100

            val = 0;

            % A(j-1,k,l)
            if j > 1
                val = val + A(j-1,k,l);
            end

            % A(j+1,k,l)
            if j < 100
                val = val + A(j+1,k,l);
            end

            % A(j,k-1,l)
            if k > 1
                val = val + A(j,k-1,l);
            end

            % A(j,k+1,l)
            if k < 100
                val = val + A(j,k+1,l);
            end

            % A(j,k,l-1)
            if l > 1
                val = val + A(j,k,l-1);
            end

            % A(j,k,l+1)
            if l < 100
                val = val + A(j,k,l+1);
            end

            % Assign computed value
            L(j,k,l) = val;
        end
    end
end

end


function L = compute_L_entries(A,p1,p2,p3)

L = zeros(length(p1),length(p2),length(p3));

% Loop over all valid j,k,l from 1 to 100
for j_p = 1:length(p1)
    for k_p = 1:length(p2)
        for l_p = 1:length(p3)

            val = 0;
            j=p1(j_p);
            k=p2(k_p);
            l=p3(l_p);

            % A(j-1,k,l)
            if j > 1
                val = val + A(j-1,k,l);
            end

            % A(j+1,k,l)
            if j < 100
                val = val + A(j+1,k,l);
            end

            % A(j,k-1,l)
            if k > 1
                val = val + A(j,k-1,l);
            end

            % A(j,k+1,l)
            if k < 100
                val = val + A(j,k+1,l);
            end

            % A(j,k,l-1)
            if l > 1
                val = val + A(j,k,l-1);
            end

            % A(j,k,l+1)
            if l < 100
                val = val + A(j,k,l+1);
            end

            % Assign computed value
            L(j_p,k_p,l_p) = val;
        end
    end
end

end








