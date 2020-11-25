%% Information
% Author: Lorenzo Bulosan
% Course: MSc Human and Biological Robotics
% Module: BE9-MMLNC

% PURPOSE:
%       Testing different architectures
%       Hyperparameters:
%                   Hidden Nodes
%                   Learning Rate
%                   #iterations training

% Setting up
clc; clear all;
load data.mat

% Splitting whole data to obtain training data
AllData = data;
AllData = AllData(randperm(size(AllData,1)),:); % Randomise
split=round(0.8*size(AllData,1));
Training_data = AllData(1:split,:);

% Splitting training data to obtain validation data
subsplit = round(0.8*size(Training_data,1));
SubTraining_data = Training_data (1:subsplit,:);
Validation_data = Training_data (1+subsplit:end,:);

% Separate input and labels
SubTraining_input = SubTraining_data(:,2:end);
SubTraining_label = SubTraining_data(:,1);
Validation_input = Validation_data(:,2:end);
Validation_label = Validation_data(:,1);


%% Hyperparameter to test: Hidden Nodes------------------------------------
%--------------------------------------------------------------------------
Record_RMSE_Train=[];
Record_RMSE_validation=[];

disp('Testing different number of hidden nodes')
for HiddenNodes = 1:20:200
   
    % Set constants other hyperparameters (alpha and #epoch)
    alpha = 0.1; % initialised guess
    epoch = 1;
    
    % Calling Modified TrainClassifierX.m 
    [parameters,RMSE_train,~] = TrainClassifierX_Hyperparameter...
        (SubTraining_input,SubTraining_label,HiddenNodes,alpha,epoch);
    
    % test architechtures with validation data
    [~,RMSE_validation]=MLP_Classifier(Validation_input,parameters,Validation_label);
     
    Record_RMSE_Train =[Record_RMSE_Train;RMSE_train];
    Record_RMSE_validation=[Record_RMSE_validation,RMSE_validation];
end

% Plot and find the best #Hidden Nodes
plot(1:20:200,Record_RMSE_Train,'k-'); hold on
plot(1:20:200,Record_RMSE_validation,'b-');
xlabel('Number of Hidden Neurons');
ylabel('RMSE');


%% Hyperparameter to test: Learning rate ----------------------------------
%--------------------------------------------------------------------------

Record2_RMSE_Train=[];
Record2_RMSE_validation=[];

disp('Testing different Learning Rates')
for alpha = 0.0001:0.005:0.1
   
    % Set constants other hyperparameters (#HiddenNodes and #epoch)
    HiddenNodes = 120; % This is what was infered from above for loop
    epoch = 1;
    
    % Calling Modified TrainClassifierX.m 
    [parameters,RMSE_train,~] = TrainClassifierX_Hyperparameter...
        (SubTraining_input,SubTraining_label,HiddenNodes,alpha,epoch);
    
    % test architechtures with validation data
    [~,RMSE_validation]=MLP_Classifier(Validation_input,parameters,Validation_label);
     
    Record2_RMSE_Train =[Record2_RMSE_Train;RMSE_train];
    Record2_RMSE_validation=[Record2_RMSE_validation,RMSE_validation];
end

% Plot and find the best Learning Rate
plot(0.0001:0.005:0.1,Record2_RMSE_Train,'k-'); hold on
plot(0.0001:0.005:0.1,Record2_RMSE_validation,'b-');
xlabel('Learning Rate alpha');
ylabel('RMSE');

%% Hyperparameter to test: Epoch ----------------------------------
%--------------------------------------------------------------------------

Record3_RMSE_Train=[];
Record3_RMSE_validation=[];

disp('Testing different training epoch')
for epoch = 1:5
    tic
    % By observing results of above loops
    HiddenNodes = 120; 
    alpha = 0.08;
    
    % Calling Modified TrainClassifierX.m 
    [parameters,RMSE_train,~] = TrainClassifierX_Hyperparameter...
        (SubTraining_input,SubTraining_label,HiddenNodes,alpha,epoch);
    
    % test architechtures with validation data
    [~,RMSE_validation]=MLP_Classifier(Validation_input,parameters,Validation_label);
     
    Record3_RMSE_Train =[Record3_RMSE_Train;RMSE_train];
    Record3_RMSE_validation=[Record3_RMSE_validation,RMSE_validation];
    toc
end

%% Plot and find the best Learning Rate
plot(1:5,Record3_RMSE_Train,'k-'); hold on
plot(1:5,Record3_RMSE_validation,'b-');
xlabel('Epoch');
ylabel('RMSE');


%% Copy of TrainClassifierX.m with hyperparameters as arguments
function [parameters,RMSE,RMSE_epoch] = TrainClassifierX_Hyperparameter(input,label,HiddenNodes,alpha,epoch)

% Same code as TrainClassifierX.m but with hyperparameters 
% (hidden nodes , learning rate, # iterations) as new arguments
% for easy changing of values when calling function.  

% Structure Neural Network
InputNodes = size(input,2);
HiddenNodes = HiddenNodes;
OutputNodes = 5;

% Initialise weights and bias from input to hidden layer
Weights_IH = rand([InputNodes,HiddenNodes]);    
Bias_IH = rand([HiddenNodes,1]);
% Initialise weights and bias from hidden layer to output
Weigths_HO = rand([HiddenNodes,OutputNodes]); 
Bias_HO = rand([OutputNodes,1]);

% Normalise data and couple it with its label to keep order 
NormInput = normalize(input,2);
AllData =[label NormInput];

% Split validation data and training data
split=round(0.9*size(AllData,1));
TrainingData = AllData(1:split,:) ;
ValidationData = AllData(1+split:end,:) ;

%% Training for n sets, every datapoint
alpha = alpha;
epoch = epoch;
box = waitbar(0,'Epoch: 1'); pause(0.5);

for i = 1:epoch
 
    % Randomize every new epoch so it's not trained on the same order
    Randomize_data = TrainingData(randperm(size(TrainingData,1)),:);
    input = Randomize_data(:,2:end);
    label = Randomize_data(:,1);  
    
        % Update parameters every data point
        for j = 1:size(input,1)
        Traindata = input(j,:)';
        %% ----------------------FeedForward----------------------- 
        % Hidden Layer output
        Hidden_layer = Weights_IH' * Traindata + Bias_IH; 
        Hidden_layer_output = ToEveryElement(Hidden_layer,'Sigmoid');
        % Output Layer output
        Output_layer = Weigths_HO' * Hidden_layer_output + Bias_HO;
        Output = ToEveryElement(Output_layer,'Sigmoid');

        %% -------------------Backpropagation----------------------

        % Finding Delta Weight of Hidden to output 
        Label_array = ToArray (label(j));      
        Output_errors = Label_array' - Output; 
        % Calculating Gradients for Delta Weights hidden to output
        Output_derrivative = ToEveryElement(Output,'dSigmoid');
        Gradients_HO = alpha.*Output_errors.*Output_derrivative;
        % Calculating Deltas of weigths from hidden to output
        Deltas_Weigths_HO = Gradients_HO * Hidden_layer_output';
        % Updating Weigths_HO
        Weigths_HO = Weigths_HO + Deltas_Weigths_HO';

        % REPEAT ABOVE to find Weights_IH
        Hidden_errors = Weigths_HO * Output_errors;
        Hidden_derrivative= ToEveryElement(Hidden_layer_output,'dSigmoid');
        Gradients_IH = alpha.*Hidden_errors.*Hidden_derrivative;
        Deltas_Weights_IH = Gradients_IH * Traindata';
        % Updating Weigths_IH
        Weights_IH = Weights_IH + Deltas_Weights_IH';

        % Updating the biases
        Bias_HO = Bias_HO + Gradients_HO;
        Bias_IH = Bias_IH + Gradients_IH;
        
        end

    % Progress information for user
    box = waitbar(i/epoch,box,strcat('Epoch:',num2str(i)));
    
    % RMSE error in this epoch
    MSE = mean(Output_errors.^2);
    RMSE_epoch(i,1)= sqrt(MSE);
end

% Final RMSE error after training
RMSE = RMSE_epoch(end);

% Clossing progress bar
close(box);

%% Trained parameters
parameters.Weights_IH = Weights_IH;    
parameters.Weigths_HO = Weigths_HO; 
parameters.Bias_IH = Bias_IH; 
parameters.Bias_HO = Bias_HO;

%% -----------------------------------------Functions made----------------------------------------
    % Apply any function to every element
    function [result] = ToEveryElement(x,func)
        for row = 1:size(x,1)
            for col = 1:size(x,2)
                switch func
                    case'Sigmoid'
                        result(row,col)= Sigmoid(x(row,col));
                    case'dSigmoid'
                        result(row,col)= dSigmoid(x(row,col));
                end
            end
        end
        return
    end

    % Sigmoid function
    function [output] = Sigmoid (x)
        output = 1/(1+exp(-x));
    end

    % Derrivative of sigmoid function
    function y = dSigmoid(input)
        y = input*(1-input);
        return
    end
    
    % Convert label to an array
    function label_array = ToArray (label)
        switch label
            case 1
                label_array = [1 0 0 0 0];
                return
            case 2
                label_array = [0 1 0 0 0];
                return
            case 3
                label_array = [0 0 1 0 0];
                return
            case 4
                label_array = [0 0 0 1 0];
                return
            case 5
                label_array = [0 0 0 0 1];
                return
        end
    end
    
end

%% Copy of ClassifyX.m
function [label,RMSE] = MLP_Classifier(input,parameters,True_Label)
%  Author: Lorenzo Bulosan 07/12/2018-start
%  For completion of module BE9-MMLNC
% Coursework 2

% Normalise input
input = normalize(input,2);

% Expand the parameters
Weights_IH = parameters.Weights_IH;
Weigths_HO = parameters.Weigths_HO;
Bias_IH = parameters.Bias_IH;
Bias_HO = parameters.Bias_HO;


for i = 1:size(input,1)
    Data = input(i,:)';
    
    %% FeedForward 
    % Hidden Layer output
    Hidden_layer = Weights_IH' * Data + Bias_IH; 
    Hidden_layer_output = ToEveryElement(Hidden_layer,'Sigmoid');
    % Output Layer output
    Output_layer = Weigths_HO' * Hidden_layer_output + Bias_HO;
    Output = ToEveryElement(Output_layer,'Sigmoid');
    
    % Predicted label
    label(i,:) = find(Output==max(Output));

end
    % RMSE
    True_Label = ToArray(True_Label(end));
    Output_errors = Output-True_Label';
    MSE = mean(Output_errors.^2);
    RMSE = sqrt(MSE);


%% ----------------------------Functions-----------------------------------
    % Apply any function to every element
    function [result] = ToEveryElement(x,func)
        for row = 1:size(x,1)
            for col = 1:size(x,2)
                switch func
                    case'Sigmoid'
                        result(row,col)= Sigmoid(x(row,col));
                    case'dSigmoid'
                        result(row,col)= dSigmoid(x(row,col));
                end
            end
        end
        return
    end

    % Sigmoid function
    function [output] = Sigmoid (x)
        output = 1/(1+exp(-x));
    end
    
    % Derrivative of sigmoid function
    function y = dSigmoid(input)
        y = input*(1-input);
        return
    end

    % Convert label to an array
    function label_array = ToArray(label)
        switch label
            case 1
                label_array = [1 0 0 0 0];
                return
            case 2
                label_array = [0 1 0 0 0];
                return
            case 3
                label_array = [0 0 1 0 0];
                return
            case 4
                label_array = [0 0 0 1 0];
                return
            case 5
                label_array = [0 0 0 0 1];
                return
        end
    end
    
%  Author: Lorenzo Bulosan  18/12/18 -end 
end




