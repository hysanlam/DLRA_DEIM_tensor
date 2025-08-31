A=randn(100,5)
U=orth(A)

[J,M] = arp(U,true)
[~,~,p3]=sRRQR_rank( U',2, 5 );
p3=p3(1:5)
