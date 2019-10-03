function [J grad] = nnCostFunction2Hidden(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer1_size, ...
                                   hidden_layer2_size, ...
                                   num_labels, ...
                                   X, y, lambda)

startOfTheta1 = 1;
endOfTheta1 = (hidden_layer1_size * (input_layer_size + 1));
sizeOfTheta1 = [hidden_layer1_size, (input_layer_size + 1)];

startOfTheta2 = (1+hidden_layer1_size * (input_layer_size+1));
endOfTheta2 = (hidden_layer1_size * (input_layer_size+1) + hidden_layer2_size * (hidden_layer1_size+1));
sizeOfTheta2 = [hidden_layer2_size, (hidden_layer1_size+1)];

startOfTheta3 = (1 + hidden_layer1_size * (input_layer_size+1) + hidden_layer2_size * (hidden_layer1_size+1));
sizeOfTheta3 = [num_labels, (hidden_layer2_size+1)];

%Reshaping Theta1, Theta2 and Theta3 from nn_params
Theta1 = reshape(nn_params( startOfTheta1:endOfTheta1 ), sizeOfTheta1);

Theta2 = reshape(nn_params( startOfTheta2:endOfTheta2 ), sizeOfTheta2);
             
Theta3 = reshape(nn_params( startOfTheta3:end ) , sizeOfTheta3);

% Theta1 = reshape(nn_params( 1:140140 ), 140, 1001);
% 
% Theta2 = reshape(nn_params( 140141:141550 ), 10, 141);
%              
% Theta3 = reshape(nn_params( 141551:end ) , sizeOfTheta3);

% Setup some useful variables
m = size(X, 1);
X = [ones(m,1) X];
         
Y = zeros(m, num_labels);

% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
Theta3_grad = zeros(size(Theta3));

%Feedforwarding of the neural network and returning the cost in the variable J.

h1_transpose = sigmoid(X*Theta1');
h1_transpose = [ones(size(h1_transpose, 1),1) h1_transpose];

h2_transpose = sigmoid(h1_transpose*Theta2');
h2_transpose = [ones(size(h2_transpose, 1),1) h2_transpose];

h = sigmoid(Theta3*h2_transpose');

for i = 1:m
    Y(i, y(i)) = 1;%5000 X 10
end

J = 1/m * (sum(sum(-Y.*log(h') - (1-Y).*log(1-h'))) + lambda/2 * (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2))));

Delta_1 = zeros(size(Theta1, 1), size(Theta1,2));
Delta_2 = zeros(size(Theta2, 1), size(Theta2,2));
Delta_3 = zeros(size(Theta3, 1), size(Theta3,2));

for t = 1:m
    a_1 = X(t, :);%1 X 1001
    z_2 = a_1*Theta1';%1 X 70
    a_2 = [1 sigmoid(z_2)];%1 X 71
    z_3 = a_2*Theta2';%1 X 5
    a_3 = [1 sigmoid(z_3)];%1 X 6
    z_4 = a_3*Theta3';%1 X 10
    a_4 = sigmoid(z_4);%row vector 1 X 10
   
    %Error Calculation for each layer
    delta_4 = a_4' - Y(t, :)'; %10 X 1 coloumn vector

    delta_3 = (Theta3')*(delta_4).*(sigmoidGradient([1 z_3]')); %(6 X 10)*(10 X 1).*(6 X 1) column vector
    delta_3 = delta_3(2:end, :); %(5 X 1) column vector
    
    delta_2 = (Theta2')*(delta_3).*(sigmoidGradient([1 z_2]')); %(71 X 5)*(5 X 1).*(71 X 1) column vector
    delta_2 = delta_2(2:end, :); %(71 X 1) column vector
    
    %Accumulation of Error
    Delta_1 = Delta_1 + delta_2*a_1;
    Delta_2 = Delta_2 + delta_3*a_2;
    Delta_3 = Delta_3 + delta_4*a_3;
    
end

reg_theta_1 = [zeros(size(Theta1, 1), 1) Theta1(:, 2:end)];
reg_theta_2 = [zeros(size(Theta2, 1), 1) Theta2(:, 2:end)];
reg_theta_3 = [zeros(size(Theta3, 1), 1) Theta3(:, 2:end)];

Theta1_grad = Delta_1 / m + lambda / m * reg_theta_1;
Theta2_grad = Delta_2 / m + lambda / m * reg_theta_2;
Theta3_grad = Delta_3 / m + lambda / m * reg_theta_3;

grad = [Theta1_grad(:) ; Theta2_grad(:) ; Theta3_grad(:)];


end
