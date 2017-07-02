function [TD,TL,ED,EL]=Crossvalidation(DX,v,i)

[m,n]=size(DX);
step=floor(m/v);

if i~= v
    startpoint=(i-1)*step+1;
    endpoint=(i)*step;
else
    startpoint=(i-1)*step+1;
    endpoint=m;
end
cv_p=startpoint:endpoint; %%%% test set position

%%%%%%%%%%%%%% test set
Test_data=DX(cv_p,2:n);
Test_lab=DX(cv_p,1);  %%%%label
%%%%%%%%%%%%%% training data
Train_data=DX(:,2:n);
Train_data(cv_p,:)='';
Train_lab=DX(:,1);
Train_lab(cv_p,:)='';
TD=Train_data;TL=Train_lab;ED=Test_data;EL=Test_lab;