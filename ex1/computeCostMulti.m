function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.


mul = transpose(theta) .* X;

h_theta = transpose(sum(transpose(mul)));

h_theta_minus_y = h_theta - y;

h_theta_minus_y_sq = h_theta_minus_y .* h_theta_minus_y;

eta = sum(h_theta_minus_y_sq);

J = eta / (2*m);



% =========================================================================

end
