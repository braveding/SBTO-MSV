function result=fitnessfun(params, X, Y, X_test, Y_test)
mm=size(params,1);
temp=zeros(mm,1);
for i=1:mm
    temp(i) = msv_fitness(params(i,:), X, Y, X_test, Y_test);
end
result=temp;
end