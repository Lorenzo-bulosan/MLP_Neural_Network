function parameters = TrainClassifierX(input,label)
%% Information
% Author: Lorenzo Bulosan
% Course: MSc Human and Biological Robotics
% Module: BE9-MMLNC

% Structure Neural Network
InputNodes = size(input,2);
OutputNodes = 5;
HiddenNodes = 120; %Found by testing different values see Q3

% Initialise weights and bias from input to hidden layer
Weights_IH = rand([InputNodes,HiddenNodes]);    
Bias_IH = rand([HiddenNodes,1]);
% Initialise weights and bias from hidden layer to output
Weigths_HO = rand([HiddenNodes,OutputNodes]); 
Bias_HO = rand([OutputNodes,1]);

% Normalise data and couple it with its label to keep order 
NormInput = normalize(input,2);
AllData =[label NormInput];

%% Training for n sets, every datapoint
epoch = 4;%Found by testing different values see Q3
box = waitbar(0,'Epoch: 1'); pause(0.5);

for i = 1:epoch
 
    % Randomize every new epoch so it's not trained on the same order
    Randomize_data = AllData(randperm(size(AllData,1)),:);
    input = Randomize_data(:,2:end);
    label = Randomize_data(:,1);
    alpha = 0.08;%Found by testing different values see Q3
    
        % UPDATE PARAMETERS EVERY DATA POINT = stochastic
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
end
close(box);

%% Trained parameters
parameters.Weights_IH = Weights_IH;    
parameters.Weigths_HO = Weigths_HO; 
parameters.Bias_IH = Bias_IH; 
parameters.Bias_HO = Bias_HO;

%% Functions made
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