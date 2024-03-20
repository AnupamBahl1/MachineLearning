function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)

m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    hx_theta = transpose(theta) * transpose(X);
    diff_theta = transpose(hx_theta) .- y;
    temp_theta_0 = theta(1, 1) - ((alpha/m) * sum(diff_theta));
    temp_theta_1 = theta(2, 1) - ((alpha/m) * sum(diff_theta .* X(:,2)));

    theta = [temp_theta_0; temp_theta_1];
    
    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
