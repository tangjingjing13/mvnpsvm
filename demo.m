tic;
clear;close all;clc;

%dataset is a struct, x named one view, x2 named another view, y is the label of sample

S=load('toy.mat');
vv1=S.x;[m1,n1]=size(vv1);
vv2=S.x2;[m2,n2]=size(vv2);
label=S.y;label(label==-1)=0;
DX1=[label vv1];clear vv1;
DX2=[label vv2];clear vv2;
rng(1,'v5uniform');
s=randperm(m1);DX1=DX1(s,:);DX2=DX2(s,:);

iter_run=5;v=5;
eps=0.01;
c=0.1;dd=0.05;hh=dd;

accuracy1=zeros(1,iter_run);err1=zeros(1,iter_run);
for iter=1:iter_run
    %disp(['The crossvalidation iteration is ',num2str(iter)]);           
    [TX1,TY1,EX1,EY1]=Crossvalidation(DX1,v,iter);
    [TX2,TY2,EX2,EY2]=Crossvalidation(DX2,v,iter);
    [mt,nt]=size(TX1);[me,ne]=size(EX1);
    
    pos1=length(find(TY1==1));neg1=length(find(TY1~=1));
    V(1).pos=[TX1(TY1==1,:) ones(pos1,1)];V(1).neg=[TX1(TY1~=1,:) ones(neg1,1)];
    V(2).pos=[TX2(TY2==1,:) ones(pos1,1)];V(2).neg=[TX2(TY2~=1,:) ones(neg1,1)];
    [mp,~]=size(V(1).pos);[mn,~]=size(V(1).neg);
    rho=1;Paras=[eps*ones(1,2) c*ones(1,2) dd c*ones(1,2) hh];
    lambdap=ADMMforMVNPSVM(V,Paras,rho);
    wpa=(lambdap(1:mp)-lambdap(mp+1:2*mp)+lambdap(5*mp+2*mn+1:6*mp+2*mn)-...
        lambdap(4*mp+2*mn+1:5*mp+2*mn))'*V(1).pos-lambdap(4*mp+1:4*mp+mn)'*V(1).neg;
    wpb=(lambdap(2*mp+1:3*mp)-lambdap(3*mp+1:4*mp)-lambdap(5*mp+2*mn+1:6*mp+2*mn)+...
        lambdap(4*mp+2*mn+1:5*mp+2*mn))'*V(2).pos-lambdap(4*mp+mn+1:4*mp+2*mn)'*V(2).neg; 
    
    temp1=V(1).pos;V(1).pos=V(1).neg;V(1).neg=temp1;
    temp2=V(2).pos;V(2).pos=V(2).neg;V(2).neg=temp2;
    [mp,~]=size(V(1).pos);[mn,~]=size(V(1).neg);
    lambdap=ADMMforMVNPSVM(V,Paras,rho);
    wna=(lambdap(1:mp)-lambdap(mp+1:2*mp)+lambdap(5*mp+2*mn+1:6*mp+2*mn)-...
        lambdap(4*mp+2*mn+1:5*mp+2*mn))'*V(1).pos-lambdap(4*mp+1:4*mp+mn)'*V(1).neg;
    wnb=(lambdap(2*mp+1:3*mp)-lambdap(3*mp+1:4*mp)-lambdap(5*mp+2*mn+1:6*mp+2*mn)+...
        lambdap(4*mp+2*mn+1:5*mp+2*mn))'*V(2).pos-lambdap(4*mp+mn+1:4*mp+2*mn)'*V(2).neg;
    
    dis1=abs([EX1 ones(me,1)]*wpa')+abs([EX2 ones(me,1)]*wpb');%
    dis2=abs([EX1 ones(me,1)]*wna')+abs([EX2 ones(me,1)]*wnb');%
    ind=(dis1<dis2);%ind(ind==0)=-1;
    cou=length(find(ind==EY1));                      
    accuracy1(iter)=cou/me*100;
end

Acc=mean(accuracy1);
