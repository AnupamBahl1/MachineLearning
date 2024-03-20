function p = predict(Theta1, Theta2, X)
% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

p = zeros(size(X, 1), 1);

X = [ones(m, 1) X];
A1 = sigmoid(X * transpose(Theta1));

A1_m = size(A1, 1);
A1 = [ones(A1_m, 1) A1];

A2 = sigmoid(A1 * transpose(Theta2));
[x, p] = max(A2, [], 2);


end
