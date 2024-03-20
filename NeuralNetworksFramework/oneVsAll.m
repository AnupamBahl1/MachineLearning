function [all_theta] = oneVsAll(X, y, num_labels, lambda)
% Some useful variables
m = size(X, 1);
n = size(X, 2);

all_theta = zeros(num_labels, n + 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];


for index = 1:num_labels
	initial_theta = zeros(n+1, 1);
	options = optimset('GradObj', 'on', 'MaxIter', 50);
	
	curr_y = y == index;

	theta = fmincg(@(t)(lrCostFunction(t, X, curr_y, lambda)), initial_theta, options);

	all_theta(index, :) = theta(:);
end;


end
