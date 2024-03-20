function [X_norm, mu, sigma] = featureNormalize(X)
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));


for index = 1:size(X,2)
	mu(:, index) = mean(X(:, index));
	sigma(:, index) = std(X(:, index));
	X_norm(:, index) = X_norm(:, index) .- mu(:, index);
	X_norm(:, index) = X_norm(:, index) ./ sigma(:, index);
end;


end
