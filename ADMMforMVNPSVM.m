function lambda=ADMMforMVNPSVM(V,Paras,rho)
% ADMM for multi-view NPSVM
% V denotes the datasets, including v views, every view contains two part
% positive class and negative class; Paras contains related parameters.

%Global constants and defaults
MAX_ITER = 50;
% ABSTOL   = 1e-4;
% RELTOL   = 1e-2;

%vd=size(V,2);
AP=V(1).pos;AN=V(1).neg;BP=V(2).pos;BN=V(2).neg;
map=size(AP,1);man=size(AN,1);mbp=size(BP,1);mbn=size(BN,1);% map=mbp,man=mbn
ep=ones(map,1);en=ones(man,1);
eps1=Paras(1);eps2=Paras(2);c1=Paras(3);c2=Paras(4);d=Paras(5);
c3=Paras(6);c4=Paras(7);h=Paras(8);
K11=kernel(AP,AP,'linear',1,1);K12=kernel(AP,AN,'linear',1,1);
K13=kernel(AN,AN,'linear',1,1);
K21=kernel(BP,BP,'linear',1,1);K22=kernel(BP,BN,'linear',1,1);
K23=kernel(BN,BN,'linear',1,1);


HP1=[K11 -K11;-K11 K11];HP2=[-K12 zeros(map,man);K12 zeros(map,man)];
HP3=[K21 -K21;-K21 K21];HP4=[zeros(mbp,mbn) -K22;zeros(mbp,mbn) K22];
HP5=[K13 zeros(man,mbn);zeros(mbn,man) K23];HP6=[K12' -K12';-K22' K22'];
HP7=[K11+K21 -K11-K21;-K11-K21 K11+K21];
GP=[HP1 zeros(2*map,2*mbp) HP2 -HP1;zeros(2*mbp,2*map) HP3 HP4 HP3;
    HP2' HP4' HP5 HP6;-HP1' HP3' HP6' HP7];

kp=[repmat(eps1*ep',1,4) -en' -en' eps2*ep' eps2*ep']';
e1=[zeros(map,4*map+2*man) diag(ones(1,map)) diag(ones(1,map))]';
CP=[repmat(c1*ep',1,4) c2*en' c2*en' d*ep' d*ep']';

TP=[e1';diag(ones(1,6*map+2*man));diag(ones(1,6*map+2*man))];
UP=diag(ones(1,13*map+4*man));
UP(map+1:7*map+2*man,map+1:7*map+2*man)=-diag(ones(1,6*map+2*man));
ccp=[d*ep' zeros(1,6*map+2*man) CP']';

HHP=GP+rho*(TP'*TP);

P=zeros(6*map+2*man,1); % P=lambdap
L=zeros(13*map+4*man,1);% L=\xi
U=zeros(13*map+4*man,1);% 
kkp=1;fval=zeros(MAX_ITER,1);sval=fval;rval=fval;
while kkp <=MAX_ITER
    Lold=L;
    
    V = U + UP*L - ccp;
    br = - kp - rho * TP'*V;   % br is  vector b in AX=b for CG
    [P, ~] = CGsolver(HHP,br);  %CG mothed to solve a linear equations Pi
    
    % update L
    L = pos(UP'*(ccp-TP*P-U));
    
    % update U
    r = TP*P + UP*L - ccp;  %r=  A*Pi+B*Gamma, used for the primal residual
    U = U + r;
    
    fval(kkp)=0.5*P'*GP*P+kp'*P;
    
    s = rho*TP'*(UP*(L- Lold));
    
    rval(kkp)  = norm(r); %record the primal residual at k
    sval(kkp)  = norm(s); %record the dual residual at k
    
        %an absolute tolerance xi^abs  and a relative tolerance xi^rel
%     eps_pri(k) = sqrt(m+2)*ABSTOL + RELTOL*max([norm(A*P), norm(B*L), norm(cc)]); 
%     eps_dual(k)= sqrt(2*m)*ABSTOL + RELTOL*norm(rho*A'*U);
        
%     if  rval(kkp) < eps_pri(kkp) && sval(kkp) < eps_dual(kkp);
%          break
%     end    
    kkp=kkp+1;
end
lambda=P;

% HN1=[K13 -K13;-K13 K13];HN2=[K12' zeros(man,map);-K12' zeros(man,map)];
% HN3=[K23 -K23;-K23 K23];HN4=[zeros(man,map) K22';zeros(man,map) -K22'];
% HN5=[K11 zeros(map);zeros(map) K21];HN6=[-K12 K12;K22 -K22];
% HN7=[K13+K23 -K13-K23;-K13-K23 K13+K23];
% GN=[HN1 zeros(2*man) HN2 -HN1;zeros(2*man) HN3 HN4 HN3;
%     HN2' HN4' HN5 HN6;-HN1' HN3' HN6' HN7];
% 
% kn=[repmat(eps1*en',1,4) -ep' -ep' eps2*en' eps2*en']';
% e2=[zeros(man,4*man+2*map) diag(ones(1,man)) diag(ones(1,man))]';
% CN=[repmat(c3*en',1,4) c4*ep' c4*ep' h*en' h*en']';
% 
% TN=[e2';diag(ones(1,6*man+2*map));diag(ones(1,6*man+2*map))];
% UN=diag(ones(1,13*man+4*map));
% UN(man+1:7*man+2*map,man+1:7*man+2*map)=-diag(ones(1,6*man+2*map));
% ccn=[h*en' zeros(1,6*man+2*map) CN']';
% 
% HHN=GN+rho*(TN'*TN);
% 
% P=zeros(6*man+2*map,1); % P=lambdan
% L=zeros(13*man+4*map,1);% L=\xi
% U=zeros(13*man+4*map,1);% 
% kkn=1;fval=zeros(MAX_ITER,1);sval=fval;rval=fval;
% while kkn <=MAX_ITER
%     Lold=L;
%     
%     V = U + UN*L - ccn;
%     br = - kn - rho * TN'*V;   % br is  vector b in AX=b for CG
%     [P, ~] = CGsolver(HHN,br);  %CG mothed to solve a linear equations Pi
%     
%     % update L
%     L = pos(UN'*(ccn-TN*P-U));
%     
%     % update U
%     r = TN*P + UN*L - ccn;  %r=  A*Pi+B*Gamma, used for the primal residual
%     U = U + r;
%     
%     fval(kkn)=0.5*P'*GN*P+kn'*P;
%     
%     s = rho*TN'*(UN*(L- Lold));
%     
%     rval(kkn)  = norm(r); %record the primal residual at k
%     sval(kkn)  = norm(s); %record the dual residual at k
%     
%         %an absolute tolerance xi^abs  and a relative tolerance xi^rel
% %     eps_pri(k) = sqrt(m+2)*ABSTOL + RELTOL*max([norm(A*P), norm(B*L), norm(cc)]); 
% %     eps_dual(k)= sqrt(2*m)*ABSTOL + RELTOL*norm(rho*A'*U);
%         
% %     if  rval(kkn) < eps_pri(kkn) && sval(kkn) < eps_dual(kkn);
% %          break
% %     end    
%     kkn=kkn+1;
% end
% lambdan=P;
end

function [x, niters] = CGsolver(A,b)
% cgsolve : Solve Ax=b by conjugate gradients
%
% Given symmetric positive definite sparse matrix A and vector b, 
% this runs conjugate gradient to solve for x in A*x=b.
% It iterates until the residual norm is reduced by 10^-6,
% or for at most max(100,sqrt(n)) iterations

n = length(b);

tol=1e-4;
maxiters=20;

normb = norm(b);
x = zeros(n,1);
r = b;
rtr = r'*r;
d = r;
niters = 0;
while sqrt(rtr)/normb > tol  &&  niters < maxiters
    niters = niters+1;
    Ad = A*d;
    alpha = rtr / (d'*Ad);
    x = x + alpha * d;
    r = r - alpha * Ad;
    rtrold = rtr;
    rtr = r'*r;
    beta = rtr / rtrold;
    d = r + beta * d;
end
end

function A = pos(A)
A(A<0)=0;
end
