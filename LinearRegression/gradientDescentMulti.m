function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
temp_theta = theta;


for iter = 1:num_iters

    hx_theta = transpose(theta) * transpose(X);
    diff_theta = transpose(hx_theta) .- y;

    for index = 1:size(X,2)
        temp_theta(index, :) = theta(index, :) - ((alpha/m) * sum(diff_theta .* X(:,index)));
    end;

    theta = temp_theta;

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
