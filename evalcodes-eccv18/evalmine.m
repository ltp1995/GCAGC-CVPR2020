clear
close all
% imgroot1='/media/litengpeng2/litengpeng/data/coco/val2017/';
% imgroot2='/media/litengpeng2/litengpeng/data/coco/train2017/';
% imname=dir([imgroot2, '*.jpg']);
% for i=1:length(imname)
%     impath=[imgroot1, imname(i).name];
%     img=imread(impath);
%     impath2=[imgroot2, imname(i).name];
%     imwrite(img, impath2);
% 
% end
% 
addpath(genpath('EvalSaliency'));
% Method = {'YangCVPR13', 'JiangICCV13', 'GongCVPR15', 'LiCVPR15', 'JiangICCV15', 'ZhangICCV15', 'TuCVPR16', 'KongECCV16', 'HuangTIP17', 'ZhangICCV17', ...
%     'WangCVPR15', 'ZhaoCVPR15', ...
%     'LeeCVPR16', 'LiuCVPR16', 'LiCVPR16', 'WangECCV16', ...
%     'HouCVPR17', 'LuoCVPR17', 'WangICCV17', 'ZhangICCV17_UCF', 'ZhangICCV17_Amulet', ...
%     'WangCVPR17', ...
%     'FaktorICCV13', 'FuTIP13', 'CaoTIP14', 'LiuSPL14', 'LiSPL15', 'JerripothulaTMM16', 'ZhangCVPR15&IJCV16', 'ZhangTNNLS16', 'ZhangICCV15&PAMI17',...
%     'CI', 'SI', 'CI+SI', 'CI+SI+S', 'CI+SI+S+D'}; 
% 
% NameList ={'GMR', 'MC', 'TLLT', 'RRWR', 'GP', 'MB+', 'MST', 'PM', 'MILP', 'SVFSal'... % 1~10
%            'LEGS', 'MCDL', ... % 11~12
%            'ELD', 'DHS', 'DCL', 'RFCN', ... % 13~16
%            'DSS', 'NLDF', 'SRM', 'UCF', 'Amulet', ... % 17~21
%            'WSS', ... % 22
%            'CSBC', 'CBCS', 'SACS', 'CSHS', 'ESMG', 'CSSCF', 'CoDW', 'DIM', 'SP-MIL',... %23~31
%            'CI', 'SI', 'CI+SI', 'CI+SI+S', 'CI+SI+S+D'}; % 32~36
% 
% CiteNameList = {'Yang13', 'Jiang13', 'Gong15', 'Li15', 'Jiang15', 'Zhang15', 'Tu16', 'Kong16', 'Huang17', 'zhang2017supervision'...
%     'Wang15', 'Zhao15', ...
%     'Lee16', 'Liu16', 'Li16', 'Wang16', ...
%     'Hou17', 'Luo17', 'Wang17b', 'Zhang17a', 'Zhang17b', ...
%     'Wang17a', ... % 22
%     'faktor2013co', 'fu2013cluster', 'cao2014self', 'liu2014co', 'li2015efficient', 'jerripothula2016image', 'ZhangIJCV16', 'ZhangTNNLS16', 'ZhangTPAMI17', ...
%     'CI', 'SI', 'CI+SI', 'CI+SI+S', 'CI+SI+S+D'}; % 32~36

% Method   = {'FaktorICCV13', 'FuTIP13', 'CaoTIP14', 'LiuSPL14', 'LiSPL15', 'JerripothulaTMM16', 'ZhangCVPR15&IJCV16', 'ZhangTNNLS16', 'ZhangICCV15&PAMI17'}; 
% 
% NameList = {'CSBC', 'CBCS', 'SACS', 'CSHS', 'ESMG', 'CSSCF', 'CoDW', 'DIM', 'SP-MIL'}; % 32~36
% 
% CiteNameList= {'faktor2013co', 'fu2013cluster', 'cao2014self', 'liu2014co', 'li2015efficient', 'jerripothula2016image', 'ZhangIJCV16', 'ZhangTNNLS16', 'ZhangTPAMI17'}; % 32~36
% Dataset =     {'iCoseg', 'Cosal2015'};





Method = {'chenjin-fine3000'}; 

NameList ={'chenjin-fine3000'}; % 32~36

CiteNameList = {'LTP'}; % 32~36
Dataset = {'iCoseg', 'Cosal2015'};
%Dataset = {'iCoseg'};

% 

EvalResultDir = [pwd '/EvalResult/'];
%% Evaluation

mkdir(EvalResultDir)
AP = nan(length(Method), length(Dataset));
F_score = nan(length(Method), length(Dataset));
SMeasure = nan(length(Method), length(Dataset));
MAE = nan(length(Method), length(Dataset));
WriteScore = nan(length(Method) , 9);
for i = 1:length(Dataset)
    TempEvalResultDir = [EvalResultDir '/' Dataset{i}];
    mkdir(TempEvalResultDir)
    for j =1:length(Method)
        disp(['DataSet:' Dataset{i} '(' num2str(i) '/3) , Method:' Method{j} '(' num2str(j) '/' num2str(length(Method)) ') '])
        SaveName = [TempEvalResultDir '/' NameList{j} '.mat'];
        if ~exist(SaveName, 'file')
            ResultDir = [pwd '/SalMapResults-mine/' Method{j} '/' Dataset{i} '/'];
            switch Dataset{i}
                case 'iCoseg'
                    GTMaskExt='*.png';
                case 'MSRC'
                    GTMaskExt='*.bmp';
                case 'Cosal2015'
                    GTMaskExt='*.png';
                case 'coco_val'
                    GTMaskExt='*.jpg';
                case 'CoSOD3k'
                    GTMaskExt='*.png';
                    
            end
            GTPath = [pwd '/datasets/' Dataset{i} '/groundtruth/'];
            try
                [MAE, ClassAP, ClassAUC, ClassFScore, AUC, AP, F_score, TPR, Precision, FPR, ClassSMeasure, SMeasure] = EvalAllResult(ResultDir, GTPath, GTMaskExt);
                save(SaveName, 'ClassAP', 'ClassAUC', 'ClassFScore', 'AUC', 'AP', ...
                    'F_score', 'TPR', 'Precision', 'FPR', 'ClassSMeasure', 'SMeasure', 'MAE');
            catch
            end
        end
        try
            Result = load(SaveName, 'AP', 'F_score', 'SMeasure', 'MAE');
            AP(j,i) = Result.AP;
            F_score(j,i) = Result.F_score;
            SMeasure(j,i) = Result.SMeasure;
            MAE (j, i) = Result.MAE;
            WriteScore(j, (i-1) * 4 +1:i * 4) = [Result.AP Result.F_score Result.SMeasure Result.MAE];
        catch
        end
    end
end

fileID = fopen('Results.txt','w');
for i = 1:length(CiteNameList)
    if i <= 10
        SettingStr = '& SI+US ';
    elseif i <= 21
        SettingStr = '& SI+FS ';
    elseif i == 22
        SettingStr = '& SI+WS ';
    elseif i <= 31
        SettingStr = '& MI+US ';
    else
        SettingStr = '';
    end
    if i <= 31
        WriteStr = sprintf('%s~/cite{%s}     %s', NameList{i}, CiteNameList{i}, SettingStr);
    else
        WriteStr = sprintf('%s     %s', NameList{i}, SettingStr);
    end
     
    for j = 1:8
        Num = WriteScore(i,j);
        if ~isnan(Num)
            NumStr = sprintf('%0.4f', round(Num, 4) );
        else
            NumStr = '  -  ';
        end
        WriteStr = [WriteStr ' & ' NumStr];
    end
    WriteStr = [WriteStr '\\'];
    fprintf(fileID, '%s\n', WriteStr);
    if i == 10 || i == 21 || i == 22 || i == 31
        fprintf(fileID, '\n********************************\n');
    end
end
fclose(fileID)
