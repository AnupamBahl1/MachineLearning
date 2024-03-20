function J = computeCost(X, y, theta)

% Initialize some useful values
m = length(y); % number of training examples
J = 0;

hx_theta = (transpose(theta) * transpose(X));
diff_sq = ((transpose(hx_theta) .- y) .^ 2);
J = (sum(diff_sq) / (2*m));

end
