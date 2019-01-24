function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the
%   cost of using theta as the parameter for linear regression to fit the
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


theta_wout0 = theta(2:end,1);

J =  ((X*theta-y)' * (X*theta-y))/(2*m)  + (lambda/(2*m))*(theta_wout0'*theta_wout0);

grad = (1/m * ( X'*( (X*theta) - y ) ) ) + (lambda/m)*theta ;

%Adjust for first element of theta not being subject to regularization
grad(1) = grad(1) - (lambda/m)*theta(1);

% =========================================================================

grad = grad(:);

end
