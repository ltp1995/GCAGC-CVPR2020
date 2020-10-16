function [MAE, SMeasure, subAP, subAUC, f_f, tpr, fpr, pre, F_tpr, F_pre, WFmeasure] = EvalResultClassResult(ResultDir, GTPath, GTMaskExt)
soddir2 = dir(GTPath);
soddir2(1:2) = [];
pre=[];
tpr=[];
fpr=[];
F_tpr=[];
F_pre=[];
MAE0=[];
subAP = zeros(1, length(soddir2), 'single');
subAUC = zeros(1, length(soddir2), 'single');
f_f = zeros(1, length(soddir2), 'single');
SMeasure = zeros(1, length(soddir2), 'single');
WFmeasure = zeros(1, length(soddir2), 'single');
for j = 1:length(soddir2)
    soddirnamewhole2=strcat(GTPath,soddir2(j).name,'/');
    gtfile= dir(fullfile(soddirnamewhole2,GTMaskExt));
    soddirnamewhole=strcat(ResultDir,soddir2(j).name,'/');
     
    sub_pre=[];
    sub_tpr=[];
    sub_fpr=[];
    f_tpr=[];
    f_pre=[];
    MAE1=[];
    %
    %
    TempSMeasure = zeros(1, length(gtfile));
    TempWFmeasure = zeros(length(gtfile),1, 'single');
    for i=1:length(gtfile)
        gt=imread([soddirnamewhole2 gtfile(i).name(1:end-3) 'png']);
        gt = logical(gt(:,:,1));
        if min(gt(:))==1
            continue
        end
        img=imread([soddirnamewhole gtfile(i).name(1:end-3) 'png']);
        TempSMeasure(i) = EvalStructureMeasure(img, gt);
        img=mat2gray(imresize(img,size(gt)));
        if size(img,3)>1
            img=img(:,:,1);
        end
%         [Precision,TPR, FPR] = QXL_ROC( img,gt , 100 );
        [Precision,TPR, FPR] = EvalROC( img,gt , 100 );
        pre = cat(1, pre, Precision);
        tpr = cat(1, tpr, TPR);
        fpr = cat(1, fpr, FPR);
        
        sub_pre= cat(1, sub_pre, Precision);
        sub_tpr = cat(1, sub_tpr, TPR);
        sub_fpr = cat(1, sub_fpr, FPR);
        TempWFmeasure(i) = WFb(img,gt);
        % %
        [~ ,TPR ,Precision] = Fmeasure( img, gt );
        % %
        mae = mean2(abs(double(logical(gt)) - img));
        
        MAE1 = [MAE1;mae];
        f_tpr = cat(2, f_tpr, TPR);
        f_pre = cat(2, f_pre, Precision);
        
        F_tpr = cat(2, F_tpr, TPR);
        F_pre = cat(2, F_pre, Precision);
        
    end
    sub_T=mean(sub_tpr,1);
    sub_F=mean(sub_fpr,1);
    sub_P=mean(sub_pre,1);
    subAP(j) = -trapz(sub_T, sub_P);
    subAUC(j) = -trapz(sub_F, sub_T);
    f_f(j) = mean(f_tpr)*mean(f_pre)/(mean(f_tpr)+0.3*mean(f_pre))*1.3;
    SMeasure(j) = mean(TempSMeasure);
    WFmeasure(j) = mean(TempWFmeasure);
    MAE0=[MAE0;MAE1];
end
    MAE=mean(MAE0);
end

function [Q]= WFb(FG,GT)
% WFb Compute the Weighted F-beta measure (as proposed in "How to Evaluate
% Foreground Maps?" [Margolin et. al - CVPR'14])
% Usage:
% Q = FbW(FG,GT)
% Input:
%   FG - Binary/Non binary foreground map with values in the range [0 1]. Type: double.
%   GT - Binary ground truth. Type: logical.
% Output:
%   Q - The Weighted F-beta score

%Check input
if (~isa( FG, 'double' ))
    error('FG should be of type: double');
end
if ((max(FG(:))>1) || min(FG(:))<0)
    error('FG should be in the range of [0 1]');
end
if (~islogical(GT))
    error('GT should be of type: logical');
end

dGT = double(GT); %Use double for computations.


E = abs(FG-dGT);
% [Ef, Et, Er] = deal(abs(FG-GT));

[Dst,IDXT] = bwdist(dGT);
%Pixel dependency
K = fspecial('gaussian',7,5);
Et = E;
Et(~GT)=Et(IDXT(~GT)); %To deal correctly with the edges of the foreground region
EA = imfilter(Et,K);
MIN_E_EA = E;
MIN_E_EA(GT & EA<E) = EA(GT & EA<E);
%Pixel importance
B = ones(size(GT));
B(~GT) = 2-1*exp(log(1-0.5)/5.*Dst(~GT));
Ew = MIN_E_EA.*B;

TPw = sum(dGT(:)) - sum(sum(Ew(GT)));
FPw = sum(sum(Ew(~GT)));

R = 1- mean2(Ew(GT)); %Weighed Recall
P = TPw./(eps+TPw+FPw); %Weighted Precision

Q = (2)*(R*P)./(eps+R+P); %Beta=1;
% Q = (1+Beta^2)*(R*P)./(eps+R+(Beta.*P));
end
