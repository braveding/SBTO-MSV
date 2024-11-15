function acc = msv_fitness(params, X, Y, X_test, Y_test)
    C = params(1);
    sigma = params(2);
    t = templateSVM('Standardize', 1, 'KernelFunction', 'RBF', 'BoxConstraint', C, 'KernelScale', sigma);
    SVM_model = fitcecoc(X, Y, 'Learners', t, 'Coding', 'onevsone');
    predicted_label = predict(SVM_model, X_test);
    confMat = confusionmat(Y_test, predicted_label);
    acc1 = sum(diag(confMat)) / sum(confMat(:));
    acc = -acc1;
%     save('Aggregation_confusion_matrix.mat', 'confMat')
end