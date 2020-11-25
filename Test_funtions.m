
%% Information
% Author: Lorenzo Bulosan
% Course: MSc Human and Biological Robotics
% Module: BE9-MMLNC

clc; clear all;
load data.mat

%% Plotting raw data
mesh(data);

%% Separate from labels and split data
AllData = data;
AllData = AllData(randperm(size(AllData,1)),:);
split=round(0.8*size(AllData,1));

Training_data = AllData(1:split,:);
Test_data = AllData(split:end,:); % +1 so not include last value of traing data

%% Calling Trainer
parameters = TrainClassifierX (Training_data(:,2:end),Training_data(:,1));

%% Testing with same training data to see error
disp('Testing with same data')
label2 = ClassifyX (Training_data(:,2:end),parameters);

[matrixconfused, accuracy ] = ConfusionMatrix(Training_data(:,1),label2)

%% Testing with unseen Data
disp('Testing with unseen data')
label = ClassifyX(Test_data(:,2:end),parameters);

[matrixconfused, accuracy ]= ConfusionMatrix(Test_data(:,1),label)

% ----------------------------------------------Functions---------------------------------------------

%% Testing the parameters on unseen data
function result = Testing(data,label)
correct = 0;
wrong = 0;
wrong_values=[];
for i = 1:length(label)
    if label(i)==data(i,1)
        correct = correct+1;
    else
        wrong = wrong+1;
        wrong_values=[wrong_values;label(i)];
    end
end

Error = correct/length(label)*100;

fprintf('Possible correct labels: %d\n',length(label));
fprintf('#Correct: %d\n',correct);
fprintf('#wrong: %d\n', wrong);
fprintf('number %d is mostly wrong:\n ',mode(wrong_values));
fprintf('I got right: %4.2f%%\n',correct/length(label)*100);
end

function [matrix,accuracy]=ConfusionMatrix(TrueLabels,PredictedLabels)

matrix = zeros(5,5);

for index = 1:length(TrueLabels)
    if TrueLabels(index)==PredictedLabels(index)
        matrix(TrueLabels(index),PredictedLabels(index)) = matrix(TrueLabels(index),PredictedLabels(index)) + 1;
    else
        matrix(TrueLabels(index),PredictedLabels(index)) = matrix(TrueLabels(index),PredictedLabels(index)) + 1;
    end
end

accuracy = 100*sum(diag(matrix))/length(TrueLabels);

end

