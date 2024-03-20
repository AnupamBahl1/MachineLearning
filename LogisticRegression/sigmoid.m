function g = sigmoid(z)
g = zeros(size(z));
g = (g .+ 1) ./ (1 .+ (e .^ (-z)));


end
