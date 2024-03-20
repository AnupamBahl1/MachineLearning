function [J, grad] = costFunction(theta, X, y)
m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));
h_theta = sigmoid(X * theta);

J = 1/m * ((-transpose(y) * log(h_theta)) - (transpose(1 .- y) * log(1 .- h_theta)));

for index = 1:size(X, 2)
	grad(index, :) = 1/m * sum((h_theta .- y) .* X(:, index));
end;



% =============================================================

end
