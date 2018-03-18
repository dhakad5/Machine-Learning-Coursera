function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n=length(theta);
sum=0;
% You need to return the following variables correctly 
J = 0;
for i=1:m
    J=J+y(i)*log(sigmoid(theta'*X(1,:)'))+(1-y(i))*log(1-sigmoid(theta'*X(1,:)')));
end
for i=2:n
    sum=sum+theta(i)^2;
end
J=-J/m+sum*lambda/(2*m);
grad = zeros(size(theta));
    sum1=0;e=2.718;
for j=2:n
    for i=1:m
    sum1=sum1+(sigmoid(theta'*X(i,:)')-y(i))*X(i,j);
    end
grad(j)=(sum1+lambda*theta(j))/m;
sum1=0;
    end
    
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================

end
