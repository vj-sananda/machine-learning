%% Machine Learning Online Class
%  Exercise 1: Linear regression with multiple variables
%
%  Instructions
%  ------------
%
%  This file contains code that helps you get started on the
%  linear regression exercise.
%
%  You will need to complete the following functions in this
%  exericse:
%
%     warmUpExercise.m
%     plotData.m
%     gradientDescent.m
%     computeCost.m
%     gradientDescentMulti.m
%     computeCostMulti.m
%     featureNormalize.m
%     normalEqn.m
%
%  For this part of the exercise, you will need to change some
%  parts of the code below for various experiments (e.g., changing
%  learning rates).
%

%% Initialization

%% ================ Part 1: Feature Normalization ================

%% Clear and Close Figures
clear ; close all; clc

fprintf('Loading data ...\n');

%Read all CSV data
%Last column is sale prices
data=csvread('senna_house_data_formatted.csv');

%Column 1 is area
%Column 2 is Number of bedrooms
%Column 3 is Number of bathrooms
%Column 4 is Year built
%Column 5 is Age (2016-Year built)
%Column 6 is Lot area (acres)
%Column 7 is Location (-1 to 1)
%Column 8 is Cul-de-sac (1 :Yes, 0:No)
%Column 9 is Sale Price

% ---------- begin inputs -------
feature_select = [1 2 3 5 6 7 8];
output_select = [9];
house = [ 3700 5 4.5 2009 7 0.21 1 0];
%% --------- end inputs ---------

X_complete = data;
m = size(X_complete,1);

percent_train = 90 ;

pred_house = [1  house(feature_select) ] ;

train_idx = randperm( m, floor(m * percent_train/100) ) ;

X_train = X_complete( train_idx ,feature_select);
y_train = X_complete( train_idx, output_select)

validation_idx = [];
for i=1:m
  if  ( !sum( train_idx == i )  )
    validation_idx = [ validation_idx i ];
  end
end

X_validation = X_complete(validation_idx,feature_select);
y_validation = X_complete(validation_idx,output_select);

X = X_train ;
y = y_train ;

% Print out some data points
%fprintf('First 10 examples from the dataset: \n');
%[X(1:10,:) y(1:10,:)]
%fprintf(' x = [%.0f %.0f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');

fprintf('Program paused. Press enter to continue.\n');
pause;

% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');

[X mu sigma] = featureNormalize(X);

% Add intercept term to X
X = [ones(size(X,1), 1) X];

%% ================ Part 2: Gradient Descent ================

% ====================== YOUR CODE HERE ======================
% Instructions: We have provided you with the following starter
%               code that runs gradient descent with a particular
%               learning rate (alpha).
%
%               Your task is to first make sure that your functions -
%               computeCost and gradientDescent already work with
%               this starter code and support multiple variables.
%
%               After that, try running gradient descent with
%               different values of alpha and see which one gives
%               you the best result.
%
%               Finally, you should complete the code at the end
%               to predict the price of a 1650 sq-ft, 3 br house.
%
% Hint: By using the 'hold on' command, you can plot multiple
%       graphs on the same figure.
%
% Hint: At prediction, make sure you do the same feature normalization.
%

fprintf('Running gradient descent ...\n');

% Choose some alpha value
alpha = 0.1;
num_iters = 40;

% Init Theta and Run Gradient Descent
theta = zeros(size(X,2), 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);

% Plot the convergence graph
%{
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');
%}

% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');

% Estimate the price of 10505 Prezia
% 3700 sqft
% ====================== YOUR CODE HERE ======================
% Recall that the first column of X is all-ones. Thus, it does
% not need to be normalized.
price = 0;
pred_house_norm = ( pred_house - [0 mu] )./ [ 1 sigma ];
price = pred_house_norm * theta;
%price = [1  (1650-mu(1))/sigma(1)  (3-mu(2))/sigma(2)  ] * theta ; % You should change this

% ============================================================

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using gradient descent):\n $%f\n'], price);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================ Part 3: Normal Equations ================

fprintf('Solving with normal equations...\n');

% ====================== YOUR CODE HERE ======================
% Instructions: The following code computes the closed form
%               solution for linear regression using the normal
%               equations. You should complete the code in
%               normalEqn.m
%
%               After doing so, you should complete this code
%               to predict the price of a 1650 sq-ft, 3 br house.
%

%% Load Data
X = X_train ;
y = y_train ;

% Add intercept term to X
X = [ones(size(X,1), 1) X];

% Calculate the parameters from the normal equation
theta = normalEqn(X, y);

% Display normal equation's result
fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', theta);
fprintf('\n');

y_error = ([ ones(size(X_validation,1),1) X_validation ] * theta )- y_validation

% Estimate the price of a 1650 sq-ft, 3 br house
% ====================== YOUR CODE HERE ======================
price = 0; % You should change this
price = pred_house * theta ; % You should change this

% ============================================================

fprintf(['Predicted price of a 3700 sq-ft house ' ...
         '(using normal equations):\n $%f\n'], price);
