%Function that maps y with index values to a binary 0,1 vector
%Each row of input vector yindex, contain the index of the
%class selected, for that training set vector
% so yindex looks like
% [  10
%     2
%     9
%   ....
% ]
% We want to map the above to
% [ 0 0 0 0 0 0 0 0 0 1  % Index 10 set to 1 rest are 0
%   0 1 0 0 0 0 0 0 0 0  % Index 2 set to 1 rest are 0
%   0 0 0 0 0 0 0 0 1 0  % Index 9 set to 1 rest are 0
%   ....
% ]
% The number of columns in the output will be the largest
% index in y
%=============================================================
function ybin =mapIndexToBinary(yindex)
  rows = size(yindex,1);
  num_labels = max(yindex); % number of columns in output
  ybin = [];
  for row=1:rows
    y = zeros(1,num_labels); % row vector
    y(yindex(row)) = 1 ;
    ybin = [ ybin ; y ];
  end
end
