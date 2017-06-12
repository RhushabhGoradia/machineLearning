function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

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

size(Xval)
size(yval)
params = [0.01 0.03 0.1 0.3 1 3 10 30];
m = size(Xval, 1);
pred = zeros(size(m, 1));
min = 1000000;
l = X;

for i = 1:size(params, 2)
  tempC = params(i);
  for j = 1:size(params, 2)
    tempMin = 0;
    tempSigma = params(j);
    model = svmTrain(X, y, tempC, @(X, l) gaussianKernel(X, l, tempSigma));
    pred = svmPredict(model, Xval);
    tempMin = mean(double((pred ~= yval)));
    if(tempMin < min)
      min = tempMin;
      C = tempC;
      sigma = tempSigma;
    endif
  endfor
endfor
C
sigma

% =========================================================================

end
