function [theta] = normalEqn(X, y)

theta = zeros(size(X, 2), 1);

theta = pinv(transpose(X) * X) * transpose(X) * y;


end
