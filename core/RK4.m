function Y=RK4(X,F,dt)

F_1=F(X);
F_2=F(X+dt./2*F_1);
F_3=F(X+dt./2*F_2);
F_4=F(X+dt*F_3);
Y=X+(dt./6)*(F_1 + 2*F_2 + 2*F_3 + F_4);
end