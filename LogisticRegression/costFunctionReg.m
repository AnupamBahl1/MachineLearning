function [J, grad] = costFunctionReg(theta, X, y, lambda)

m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));

[J_reg, grad_reg] = costFunction(theta, X, y);
lambda_m = lambda/m;

J = J_reg + (lambda_m/2 * sum(theta(2:end, :) .^ 2));

grad(1, :) = grad_reg(1, :);

for index = 2:size(grad_reg, 1)
	grad(index, :) = grad_reg(index, :) + (lambda_m * theta(index, :));



% =============================================================

end
