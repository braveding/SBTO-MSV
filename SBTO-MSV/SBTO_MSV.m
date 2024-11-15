meanAccuracy = 0;
while meanAccuracy > -0.90
clear
clc
close all
load IEC-TC10-1
X1 = data(:, 1:end-1);
Y1 = data(:,end);

% 标准化
mean_data = mean(X1,1);
std_data = std(X1);
standardized_data = (X1 - mean_data) ./ std_data;
X1 = standardized_data;

% Create a partition object to divide the dataset
cv = cvpartition(size(X1,1), 'Holdout', 0.2);
accuracy = zeros(cv.NumTestSets, 1);
for r = 1:cv.NumTestSets
    train_indices = cv.training(r);
    test_indices = cv.test(r);
    X = X1(train_indices, :);
    Y = Y1(train_indices, :);
    X_test = X1(test_indices, :);
    Y_test = Y1(test_indices, :);
    
save('dataset.mat', 'X', 'Y', 'X_test', 'Y_test')

        Max_iter=200;     % 最大迭代次数
        dim=2;              % 问题维度
        population_no=50;   % 种群数量
        ub=[500,20];          % 搜索上限
        lb=[0.01,0.01];        % 搜索下限

        his_p1=[];
        %%初始化
        position_bird=rand(population_no, dim) .* repmat(ub - lb, population_no, 1) + repmat(lb, population_no, 1);
        fitness=fitnessfun(position_bird, X, Y, X_test, Y_test);
        history_fitness=fitness;                         
        history_position=position_bird;

        N=40;
        M=population_no-N;
        global_fitness=[];

        %%迭代寻优计算
        for iter=1:1:Max_iter
            [fitness]=fitnessfun(position_bird, X, Y, X_test, Y_test);     
            update_vec1=fitness<history_fitness;
            update_vec2=history_fitness<=fitness;
            history_fitness=update_vec1.*fitness+update_vec2.*history_fitness;   

            for k=1:1:dim
                history_position(:,k)=update_vec1.*position_bird(:,k)+update_vec2.*history_position(:,k);
            end

            [nest_fitness,nest_num]=sort(history_fitness);   
            nest_no=nest_num(1:N);               
            nest_fitness=nest_fitness(1:N);     
            nest_position=history_position(nest_no(1:N),:);  
            fly_no=nest_num(N+1:population_no);              

            best_position=nest_position(1,:);   
            best_fitness=nest_fitness(1,:);      

            global_fitness=[global_fitness;best_fitness];

            max_fitness=max(history_fitness);    
            min_fitness=min(history_fitness);

            for j=1:1:N
                for p=1:1:dim
                    d=norm(history_position(nest_no(j),p)-history_position(nest_no(:),p));
                    step1=(d/N);
                    position_bird(nest_no(j),p)=history_position(nest_no(j),p)+step1*exp(-iter/Max_iter)*cos(2*pi*(-1+2*rand()));
                    position_bird(nest_no(j),p)=max(position_bird(nest_no(j),p),lb(p));  
                    position_bird(nest_no(j),p)=min(position_bird(nest_no(j),p),ub(p));   
                end
            end

            for j=1:1:M
                a1=floor(N*rand()+1);
                C1=exp(-50*(iter/Max_iter)^2);  
                for p=1:1:dim
                    C2=rand();
                    KKK=rand();
                    if KKK>=0.5
                        position_bird(fly_no(j),p)=best_position(p)+C1*(C2*(best_position(p)-nest_position(a1,p))+nest_position(a1,p));
                    end
                    if KKK<0.5
                        position_bird(fly_no(j),p)=best_position(p)-C1*(C2*(best_position(p)-nest_position(a1,p))+nest_position(a1,p));
                    end
                    position_bird(fly_no(j),p)=max(position_bird(fly_no(j),p),lb(p));   
                    position_bird(fly_no(j),p)=min(position_bird(fly_no(j),p),ub(p));    
                end
            end
            his_p1=[his_p1;position_bird(1,:)];
        end
    BestFitness = global_fitness;
    accuracy(r,1) = BestFitness(end,:);
    %%参数求解结果
    parameters = best_position;
    svm_C = parameters(1,1);
    svm_sigma = parameters(1,2);
    fprintf('SVM参数求解结果：\n    C:%.4f;  sigma:%.4f\n',svm_C,svm_sigma);

    %%算法收敛曲线图
    figure
    ki=1:1:iter;
    plot(ki,-global_fitness)
    xlabel('Iterations')
    ylabel('Acc(%)')
    ylim([0 1])
    legend('SBTO-MSV')
    title('fitness curve')
    set(gca,'FontName','Times New Roman');
end
meanAccuracy = mean(accuracy);
fprintf('Mean Accuracy: %.2f\n', meanAccuracy);
save('result.mat', 'accuracy', 'meanAccuracy', 'parameters')
end
