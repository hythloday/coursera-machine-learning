function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


hox = sigmoid(transpose(sum(transpose(transpose(theta) .* X))));

J = (sum((-y .* log(hox)) - (1-y) .* log(1 - hox)) / m) + (lambda * sum(theta(2:end) .* theta(2:end)) / (2 * m));

grad = (transpose(sum(X .* (hox - y))) ./ m) + [0; lambda .* theta(2:end) ./ m ];


% =============================================================

end
