function k = kernel(x,y,ker,p1,p2)
% Return the kernel matrix of the inputs x,y;
% function k = kernel(x, y);
%
%	x: inputs; (Lx,N) with Lx: number of points; N: dimension
%	y: inputs; (Ly,N) with Ly: number of points
%	k: kernel matrix with dim of (Lx,Ly);

% for i=1:size(x,1)
%     for j=1:size(y,1);
%        k(i,j)= subkernel(x(i,:),y(j,:),ker,p1,p2);
%     end
% end
%
% function k=subkernel(x,y,ker,p1,p2)
k = x*y';
switch lower(ker)
    case 'linear'

    case 'poly'
%         gamma=size(x,2)/20;
%         k=k/gamma;

        k = (k + p1).^p2;
    case 'rbf'          %%%k(x,y)=exp{-p1*||x-y||^2}+p2;
        [Lx,N] = size(x);
        [Ly,N] = size(y);
        k = 2*x*y';
        k = k-sum(x.^2,2)*ones(1,Ly);
        k = k-ones(Lx,1)*sum(y.^2,2)';
        % %% k=2*x*y'-sum(x.^2,2)*ones(1,Ly)-ones(Lx,1)*sum(y.^2,2)';
        % k = exp(k/(p1^2))+p2;  %%%k(x,y)=exp{-||x-y||^2/(p1^2)}+p2;
        k=exp(p1*k)+p2; %% Here p1 is the same to gamma in the Libsvm.
        %%% More smaller, more nonlinear!!!
    case 'sigmoid'
        k = tanh(p1*k/length(x) + p2);
end

