function Score = EvalStructureMeasure(Prediction, GT)
Prediction=imresize(Prediction,size(GT));
d_Prediction = double(Prediction);
if (max(d_Prediction(:))==255)
    d_Prediction = d_Prediction./255;
end
d_Prediction = reshape(mapminmax(d_Prediction(:)',0,1),size(d_Prediction));

Score = StructureMeasure(d_Prediction, logical(GT(:,:,1)));

end

