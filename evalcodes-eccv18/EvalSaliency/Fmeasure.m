function [F, TPR, Precision] = Fmeasure( image, hsegmap )

img=mat2gray(imresize(image,[size(hsegmap,1),size(hsegmap,2)]));

hsegmap=mat2gray(hsegmap);
hsegmap=hsegmap(:,:,1);
positiveset  = hsegmap; 
negativeset = ~hsegmap ;
P=sum(positiveset(:));
N=sum(negativeset(:));%


    T=mean(img(:))+std(img(:),0);                %%%%%%%%%%%%%%%%%zhu
    %T= 2*mean(img(:));
    T=min(T,0.8);
      positivesamples = img >= T;


      TPmat=positiveset.*positivesamples;
      FPmat=negativeset.*positivesamples;
      
       PS=sum(positivesamples(:));

      TP=sum(TPmat(:));
      FP=sum(FPmat(:));

      TPR=TP/P;
      FPR=FP/N;
      Precision=TP/PS;
      if PS==0
          F=0;
          Precision=0;
          TPR=0;
      elseif TPR==0
          F=0;
      else
         F=TPR*Precision/(TPR+0.3*Precision)*1.3;
      end

end
