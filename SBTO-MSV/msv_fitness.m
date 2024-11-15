function acc = msv_fitness(params, X, Y, X_test, Y_test)
    C = params(1);
    sigma = params(2);
    t = templateSVM('Standardize', 1, 'KernelFunction', 'RBF', 'BoxConstraint', C, 'KernelScale', sigma);
    SVM_model = fitcecoc(X, Y, 'Learners', t, 'Coding', 'onevsone');
    tStart = tic;
    predicted_label = predict(SVM_model, X_test);
    inferenceTime = toc(tStart);
    disp(['Inference Time (s): ', num2str(inferenceTime)]);
    
    confMat = confusionmat(Y_test, predicted_label);
    acc1 = sum(diag(confMat)) / sum(confMat(:));
    acc = -acc1;
    
    modelSize = whos('SVM_model');
    memoryUsage = modelSize.bytes / 1024; % 转为KB
    disp(['Model Memory Usage (KB): ', num2str(memoryUsage)]);
    expected = sum(confMat, 2) * sum(confMat, 1) / sum(confMat(:)); 
    chi2_stat = sum((confMat - expected).^2 ./ expected, 'all');
    df = (size(confMat, 1) - 1) * (size(confMat, 2) - 1);
    p_value = 1 - chi2cdf(chi2_stat, df); % p-value
    disp(['p-value: ', num2str(p_value)]);
    save('confusion_matrix.mat', 'confMat', 'inferenceTime', 'memoryUsage', 'p_value')
end