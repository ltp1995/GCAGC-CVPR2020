function [MAE2, ClassAP, ClassAUC, ClassFScore, AUC, AP, F_score, T, P, F, ClassSMeasure, SMeasure, ClassWFmeasure, WFmeasure] = EvalAllResult(ResultDir, GTPath, GTMaskExt)
[MAE, ClassSMeasure, ClassAP, ClassAUC, ClassFScore, tpr, fpr, pre, F_tpr, F_pre, ClassWFmeasure] = EvalResultClassResult(ResultDir, GTPath, GTMaskExt);
T=mean(tpr,1);
F=mean(fpr,1);
P=mean(pre,1);
MAE2 =mean(MAE, 1);
AUC = -trapz(F, T);
AP = -trapz(T, P);
F_score = mean(F_tpr)*mean(F_pre)/(mean(F_tpr)+0.3*mean(F_pre))*1.3;
SMeasure = mean(ClassSMeasure);
WFmeasure = mean(ClassWFmeasure);
end