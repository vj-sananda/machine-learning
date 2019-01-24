function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and
%   sigma. You should complete this function to return the optimal C and
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 0;
sigma = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example,
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using
%        mean(double(predictions ~= yval))
%
% Matlab block comment

%{

sdfsdf
blah blah

%}

%After finding optimum values for C and sigma, hard coded them for submission
%{
Values = [ 0.01 0.03 0.1 0.3 1 3 10 30 ];
%Values = [ 0.01 0.03 ];

result = [] ;

for i = 1:length(Values)

  for j = 1:length(Values)

    C = Values(i);

    sigma = Values(j);

    model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));

    predictions = svmPredict(model, Xval);

    err = mean(double(predictions ~= yval));

    disp([C sigma err]);

    result = [result ; C sigma err] ;

  end

end

[min, idx] = min( result(:,3) ) ; % Column 3 has the error
result

disp("Minimum error for C , sigma values");
disp(result(idx,:));

C = result( idx, 1);
sigma = result(idx,2);

%}

C = 0.3;
sigma = 0.1;


% =========================================================================

end
