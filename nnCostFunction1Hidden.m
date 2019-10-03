function [J grad] = nnCostFunction1Hidden(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
m = size(X, 1);
X = [ones(m,1) X];
         
Y = zeros(m, num_labels);
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
hidden_layer_transpose = sigmoid(X*Theta1');
hidden_layer_transpose = [ones(size(hidden_layer_transpose, 1), 1) hidden_layer_transpose];

h = sigmoid(Theta2*hidden_layer_transpose');

for i = 1:m
    Y(i, y(i)) = 1;
end

J = 1/m * (sum(sum(-Y.*log(h') - (1-Y).*log(1-h')))+ lambda/2 * (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2))));

%The Feedforward and Back propagation Algorithm
Delta_1 = zeros(size(Theta1, 1), size(Theta1,2));
Delta_2 = zeros(size(Theta2, 1), size(Theta2,2));

for t = 1:m
    a_1 = X(t, :);
    z_2 = a_1*Theta1';
    a_2 = [1 sigmoid(z_2)];
    z_3 = a_2*Theta2';
    a_3 = sigmoid(z_3);
    
    delta_3 = a_3' - Y(t, :)';
    delta_2 = (Theta2')*(delta_3).*(sigmoidGradient([1 z_2]'));
    delta_2 = delta_2(2:end, :);
    
    Delta_1 = Delta_1 + delta_2*a_1;
    Delta_2 = Delta_2 + delta_3*a_2;
end
reg_theta_1 = [zeros(size(Theta1, 1), 1) Theta1(:, 2:end)];
reg_theta_2 = [zeros(size(Theta2, 1), 1) Theta2(:, 2:end)];

Theta1_grad = Delta_1 / m + lambda / m * reg_theta_1;
Theta2_grad = Delta_2 / m + lambda / m * reg_theta_2;
% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];
end
