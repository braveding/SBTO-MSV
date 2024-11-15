clear
clc
load dataset

XTrain = X;
YTrain = Y;
XTest = X_test;
YTest = Y_test;

% 训练多分类SVM模型
t = templateSVM('Standardize', 1, 'KernelFunction', 'RBF');
SVMModel = fitcecoc(XTrain, YTrain, 'Learners', t, 'Coding', 'onevsone'); 

tStart = tic;
YPred = predict(SVMModel, XTest);
inferenceTime = toc(tStart); 
disp(['Inference Time (s): ', num2str(inferenceTime)]);

% 计算混淆矩阵
confMat = confusionmat(YTest, YPred);
disp('Confusion Matrix:');
disp(confMat);

% 计算精度、召回率等指标
numClasses = numel(unique(Y));
precision = zeros(numClasses, 1);
recall = zeros(numClasses, 1);
f1Score = zeros(numClasses, 1);
for i = 1:numClasses
    TP = confMat(i, i);
    FP = sum(confMat(:, i)) - TP;
    FN = sum(confMat(i, :)) - TP;
    precision(i) = TP / (TP + FP);
    recall(i) = TP / (TP + FN);
    f1Score(i) = 2 * (precision(i) * recall(i)) / (precision(i) + recall(i));
end
accuracy = sum(diag(confMat)) / sum(confMat(:));
meanPrecision = mean(precision);
meanRecall = mean(recall);
meanf1Score = mean(f1Score);

disp('Precision for each class:');
disp(precision);
disp('Recall for each class:');
disp(recall);
disp('F1-Score for each class:');
disp(f1Score);

disp(['Average Precision: ', num2str(meanPrecision)]);
disp(['Average Recall: ', num2str(meanRecall)]);
disp(['Average f1Score: ', num2str(meanf1Score)]);
disp(['Overall Accuracy: ', num2str(accuracy)]);

% 模型内存占用
modelSize = whos('SVMModel');
memoryUsage = modelSize.bytes / 1024;     % 转为KB
disp(['Model Memory Usage (KB): ', num2str(memoryUsage)]);

% 推理时间
disp(['Inference Time (s): ', num2str(inferenceTime)]);
expected = sum(confMat, 2) * sum(confMat, 1) / sum(confMat(:)); 
chi2_stat = sum((confMat - expected).^2 ./ expected, 'all');
df = (size(confMat, 1) - 1) * (size(confMat, 2) - 1); 
p_value = 1 - chi2cdf(chi2_stat, df); % p-value
disp(['p-value: ', num2str(p_value)]);
% save('SVM_result.mat', 'meanPrecision','meanRecall', 'meanf1Score', 'accuracy', 'confMat', 'inferenceTime', 'memoryUsage', 'p_value')

