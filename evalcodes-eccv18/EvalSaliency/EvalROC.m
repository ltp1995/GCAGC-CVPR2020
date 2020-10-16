function [Precision, TPR, FPR] = EvalROC( image, hsegmap, NT )
img=mat2gray(image);

hsegmap=logical(hsegmap(:,:,1));%
img=mat2gray(imresize(img,size(hsegmap)));
img=(img*(NT-1));


targetHist = histc(img(hsegmap), 0:NT);
nontargetHist = histc(img(~hsegmap), 0:NT);
targetHist = flipud(targetHist);
nontargetHist = flipud(nontargetHist);
targetHist = cumsum( targetHist );
nontargetHist = cumsum( nontargetHist );
Precision = flipud(targetHist ./ (targetHist + nontargetHist + eps));
Precision(end) = 1;
Precision = [0, Precision'];
TPR = flipud(targetHist / sum(hsegmap(:))); % true positive
TPR(end) = 0;
TPR = [1, TPR'];
FPR = flipud(nontargetHist / sum(hsegmap(:) == 0));
FPR(end) = 0;
FPR = [1, FPR'];


end
