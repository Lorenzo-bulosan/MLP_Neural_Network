function label = ClassifyX(input,parameters)
%% Information
% Author: Lorenzo Bulosan
% Course: MSc Human and Biological Robotics
% Module: BE9-MMLNC

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
%     label(i,:) = Output;
end

return

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
    
%  Author: Lorenzo Bulosan  18/12/18 -end 
end