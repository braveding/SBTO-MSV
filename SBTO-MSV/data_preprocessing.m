% Input your XLSX file
xlsx_file = 'IEC-TC10-1.xlsx';
% Read data from the XLSX file
data = xlsread(xlsx_file);
feature_data = data(:,1:5);
label = data(:, 6);
row_sum = sum(feature_data, 2);
new_feature_data = feature_data ./ row_sum;
data = [new_feature_data, label];

% Save the data to MAT file
save('IEC-TC10-1.mat', 'data');
