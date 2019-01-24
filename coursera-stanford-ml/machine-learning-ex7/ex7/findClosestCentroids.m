function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%
m = size(X,1); % Number of training examples
norm_matrix = [];

for i=1:K  
  c = centroids(i,:); % Get the i th centroid
  mu = repmat(c,m,1); % Repeat it along the row for the next operation
  x_minus_mu = X - mu ; % Subtract centroid from every row
  
  norm = [] ;
  for j=1:m
    norm_j = x_minus_mu(j,:) * x_minus_mu(j,:)' ;
    norm = [ norm ; norm_j ];
  end
  
  norm_matrix = [ norm_matrix  norm ];
 end
 
 % Need the index (from 1 to K) which determines which 
 % centroid each training example is closest to
 % tmp holds the min value, which we don't need.
 [tmp  idx] = min( norm_matrix, [] , 2);

% =============================================================

end

