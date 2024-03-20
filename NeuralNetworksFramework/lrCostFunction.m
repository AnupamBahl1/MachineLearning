function [J, grad] = lrCostFunction(theta, X, y, lambda)

% Initialize some useful values
m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));

%fprintf("%f, %f\n", size(X));
%fprintf("%f, %f\n", size(y));

h_theta = sigmoid(X * theta);

J = 1/m * ((-transpose(y) * log(h_theta)) - (transpose(1 .- y) * log(1 .- h_theta)));
grad = 1/m * (transpose(X) * (h_theta .- y));


% Regularizing cost and gradients

lambda_m = lambda/m;
temp_theta = theta;
temp_theta(1) = 0;

J = J + (lambda_m/2 * sum(theta(2:end, :) .^ 2));

% We need to modify theta for grad because of matrix addition requiring same dimensions as grad. So we cannot use theta(2:end, :) since it lacks one column
grad = grad + (lambda_m * temp_theta);


end
