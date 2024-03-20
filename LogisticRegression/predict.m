function p = predict(theta, X)

m = size(X, 1); % Number of training examples

p = sigmoid(X * theta);

for index = 1:m
	if (p(index, :) >= 0.5)
		p(index, :) = 1;
	else
		p(index, :) = 0;
	endif
end;


end
