function [Precision,TPR, FPR, AUC,AP,F] = QXL_ROC( image, hsegmap, NT )

img=mat2gray(image);

hsegmap=mat2gray(hsegmap);%
hsegmap=hsegmap(:,:,1);
img=mat2gray(imresize(img,size(hsegmap)));
img=(img*(NT-1));

positiveset  = hsegmap; %
negativeset = ~hsegmap ;%
P=sum(positiveset(:));%
N=sum(negativeset(:));%

TPR=zeros(1,NT);
FPR=zeros(1,NT);

Precision=zeros(1,NT);
F=zeros(1,NT);

TPR(1)=1;
FPR(1)=1;
TPR(NT+2)=0;
FPR(NT+2)=0;
Precision(1)=0;
Precision(NT+2)=1;


for i=1:NT+1

      T=i-1;

      positivesamples = img >= T;

      

      TPmat=positiveset.*positivesamples;
      FPmat=negativeset.*positivesamples;
      
       PS=sum(positivesamples(:));
       if PS~=0       

      TP=sum(TPmat(:));
      FP=sum(FPmat(:));

      TPR(i+1)=TP/P;
      FPR(i+1)=FP/N;
      
      Precision(i+1)=TP/PS;
      F(i+1)=TP*Precision/(TP+0.3*Precision)*1.3;
       end
end


AUC = -trapz(FPR, TPR);
AP = -trapz(TPR, Precision);

F=mean(F(2:end-1));
end
