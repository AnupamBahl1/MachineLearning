function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));


% -------------------------------------------------------------

% COST CALCUATION

% To calculate cost, we first need h_theta (hypothesis result). We calculate that by
% traversing all the layers of the neural network. H_theta is the theta value of last layer

A1 = [ones(m, 1) X];
Z2 = A1 * transpose(Theta1);
A2 = sigmoid(Z2);

A2 = [ones(size(A2, 1), 1), A2];
Z3 = A2 * transpose(Theta2);
A3 = sigmoid(Z3);

h_theta = A3;


% Convert Y to a matrix where we represent y=3 as [0, 0, 1, 0, 0, 0, 0, 0, 0, 0](upto length k)
% Matrix multiplication yields multiplying each y(i) with each row/column of h(theta). What we need to
% do is, multiply y(i) with only the corresponding column in h_theta.

y_vector = [];
zero_labels = 1:1:num_labels;

for index=1:m
    % Yields y(i) in the required vector form, zeroes everywhere, 1 at the label location
    y_vector = zero_labels == y(index);

    % Gives current row of h_theta
    h_theta_index = h_theta(index, :);

    J += (-(y_vector) * transpose(log(h_theta_index))) - ((1 .- y_vector) * transpose(log(1 .- h_theta_index)));    
end;

J = J/m;


% -------------------------------------------------------------




% -------------------------------------------------------------

% REGULARIZATION OF COST J

% Removing first column of thetas of all layers. It corresponds to the bias unit.
Theta1_R = Theta1(:, 2:end);
Theta2_R = Theta2(:, 2:end);

R_Cost = (lambda / (2*m)) * (sum(sum(Theta1_R .^ 2)) + sum(sum(Theta2_R .^ 2)));

J += R_Cost;


% -------------------------------------------------------------




% -------------------------------------------------------------

% GRADIENT CALCULATION

for index=1:m
    % Forward Propagation
    y_vector = zero_labels == y(index);

    a_1 = [1 X(index, :)];
    z_2 = a_1 * transpose(Theta1);
    a_2 = sigmoid(z_2);

    a_2 = [1 a_2];
    z_3 = a_2 * transpose(Theta2);
    a_3 = sigmoid(z_3); 


    % Backward propagation. Calculate deltas and gradients
    delta_3 = a_3 .- y_vector;
    delta_2 = (transpose(Theta2) * transpose(delta_3)) .* sigmoidGradient(transpose([1 z_2]));

    
    Theta2_grad = Theta2_grad + (transpose(delta_3) * a_2);
    Theta1_grad = Theta1_grad + (delta_2(2:end) * a_1);
end;


Theta1_grad = Theta1_grad ./ m;
Theta2_grad = Theta2_grad ./ m;

% -------------------------------------------------------------




% -------------------------------------------------------------

% REGULARIZATION OF GRADIENTS

Theta_Zeroes = Theta1(:, 2:end);
Theta_Zeroes = [zeros(size(Theta_Zeroes, 1), 1) Theta_Zeroes];

Theta1_grad = Theta1_grad .+ ((lambda / m) .* Theta_Zeroes);

Theta_Zeroes = Theta2(:, 2:end);
Theta_Zeroes = [zeros(size(Theta_Zeroes, 1), 1) Theta_Zeroes];

Theta2_grad = Theta2_grad .+ ((lambda /m) .* Theta_Zeroes);


% -------------------------------------------------------------



function printData(mat, name)
    fprintf("%s size : %f, %f\n", name, size(mat));
end;



% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
